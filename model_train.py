import gc
import random
from itertools import cycle
import torch
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    get_cosine_schedule_with_warmup
)
from Gemma3InfiniAttention import Gemma3WithInfiniAttention
from CustomDataset import CustomDataset

from const import CONFIG, CONST
from utils import load_checkpoint

device = "cuda" if torch.accelerator.is_available() else "cpu"

torch._inductor.config.max_autotune_gemm = False
torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

def get_data_loaders():
    _data_blocks = list(range(CONST.TOKENIZED_FILE_COUNT))
    random.shuffle(_data_blocks)

    def split_by_ratio(lst, ratio):
        idx = int(len(lst) * ratio)
        return lst[:idx], lst[idx:]

    _train_datablock, _test_datablock = split_by_ratio(
        _data_blocks, CONFIG.train_test_ratio
    )
    _train_dataset = CustomDataset(_train_datablock)
    _test_dataset = CustomDataset(_test_datablock)
    _shuffled_train_dataset = ShufflerIterDataPipe(
        IterableWrapper(_train_dataset), buffer_size=CONFIG.buffer_size
    )
    _shuffled_test_dataset = ShufflerIterDataPipe(
        IterableWrapper(_test_dataset), buffer_size=CONFIG.buffer_size
    )
    train_dataloader = DataLoader(
        _shuffled_train_dataset,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.worker_count,
    )
    test_dataloader = DataLoader(
        _shuffled_test_dataset,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.worker_count,
    )
    train_dataloader = cycle(train_dataloader)
    test_dataloader = cycle(test_dataloader)

    return train_dataloader, test_dataloader


def training_step(device, model, dataloader, optimizer, scheduler, steps):
    model.train()
    optimizer.zero_grad()
    losses = 0.0
    loss_steps = []
    last_loss = 0.0
    for step in tqdm(range(steps)):
        step_loss = 0.0
        for _ in range(CONFIG.gradient_accumulation_step):
            tokens, attn_mask = next(dataloader)
            tokens = tokens.to(model.device)
            attn_mask = attn_mask.to(model.device)
            labels = tokens.clone()
            labels[:, :-1] = tokens[:, 1:]
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss = model.computeLossForTraining(
                    tokens,
                    attn_mask,
                    labels,
                    gradient_accumulation_step=CONFIG.gradient_accumulation_step,
                    chunk_size=CONFIG.chunk_size,
                )
            step_loss += loss
            del tokens, attn_mask, labels, loss

        last_loss = step_loss
        losses += step_loss
        loss_steps.append(step_loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print(f"Training loss: AVG {losses / steps}, LAST {last_loss}")
    return loss_steps


def testing_step(model, dataloader, steps):
    model.eval()
    losses = 0
    for step in range(steps):
        tokens, attn_mask = next(dataloader)
        tokens = tokens.to(model.device)
        attn_mask = attn_mask.to(model.device)
        labels = tokens.clone()
        labels[:, :-1] = tokens[:, 1:]
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            with torch.no_grad():
                loss = model.computeLossForTesting(
                    input_ids=tokens,
                    attention_mask=attn_mask,
                    target=labels,
                )
        losses += loss

    print(f"Test loss: {losses / steps}")


def main():
    print("Using device: ", device)

    train_dataloader, test_dataloader = get_data_loaders()

    model = Gemma3WithInfiniAttention(CONFIG.beta, CONFIG.segment_length).to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, CONFIG.warmup_step, CONFIG.train_step * CONFIG.epoch
    )
    print(model)

    start_epoch, losses = load_checkpoint(device, model, optimizer, scheduler)
    epoch = start_epoch
    model.compile()
    while epoch < CONFIG.epoch + 1:
        print(f"Epoch {epoch}/{CONFIG.epoch}")
        loss_steps = training_step(
            device,
            model,
            train_dataloader,
            optimizer,
            scheduler,
            CONFIG.train_step,
        )
        losses.extend(loss_steps)

        gc.collect()
        torch.cuda.empty_cache()

        testing_step(model, test_dataloader, CONFIG.test_steps)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": losses,
            },
            CONST.CHECKPOINT_PATH + f"checkpoint_epoch_{epoch}.pth",
        )
        epoch += 1
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
