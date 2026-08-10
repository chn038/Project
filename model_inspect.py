from const import CONFIG, CONST
from utils import load_checkpoint, getTestPrompt, getRandomPrompt
from Gemma3InfiniAttention import Gemma3WithInfiniAttention
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
import torch
import pandas

device = "cuda" if torch.accelerator.is_available() else "cpu"


def export_loss(losses):

    loss_dataframe = pandas.DataFrame({"loss": losses})

    print(loss_dataframe.head())
    loss_dataframe.to_csv("losses.csv", index=False)
    print("Loss csv exported")

def inspect_beta(model):
    betas = []
    for layer in model.original_model.model.layers:
        beta = layer.self_attn.beta.detach().cpu()
        betas.append(torch.sigmoid(beta).mean().item())

    print(betas)
    betas_df = pandas.DataFrame({'beta': betas})
    betas_df.to_csv("beta.csv")


def inspect_memory(model, input_ids, attention_mask):
    layers_memories = [[] for _ in range(len(model.original_model.model.layers))]
    layers_z = [[] for _ in range(len(model.original_model.model.layers))]
    layers_avg = [[] for _ in range(len(model.original_model.model.layers))]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    model = model.to(device)

    print("Passing dummy prompt...")
    model.eval()
    model._clear_all_memories()
    segments = model._segment_input(input_ids, attention_mask)
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            for segment in segments:
                segment_input_ids, segment_attention_mask = segment
                for i, layer in enumerate(model.original_model.model.layers):
                    hid, z = layer.self_attn.hid_storage.getMemory()
                    if hid is not None and z is not None:
                        layers_memories[i].append(hid.abs().max().item())
                        layers_z[i].append(z.abs().max().item())
                        layers_avg[i].append(hid.abs().max().item() / z.abs().max().item())
                    else:
                        layers_memories[i].append(0)
                        layers_z[i].append(1)
                        layers_avg[i].append(0)

                _ = model.original_model.model(
                    input_ids=segment_input_ids, attention_mask=segment_attention_mask
                )[0]
                model._manual_update_memory()
        model._clear_all_memories()

    print("Exporting...")
    layers_memories_T = [list(row) for row in zip(*layers_memories)]
    layers_z_T = [list(row) for row in zip(*layers_z)]
    layers_avg_T = [list(row) for row in zip(*layers_avg)]
    memories_df = pandas.DataFrame(layers_memories_T, columns=[f"layer{i}" for i in range(len(model.original_model.model.layers))])
    z_df = pandas.DataFrame(layers_z_T, columns=[f"layer{i}" for i in range(len(model.original_model.model.layers))])
    avg_df = pandas.DataFrame(layers_avg_T, columns=[f"layer{i}" for i in range(len(model.original_model.model.layers))])
    memories_df.to_csv("memories.csv")
    z_df.to_csv("z.csv")
    avg_df.to_csv("avg.csv")
    print("Exported")




def main():
    print("Using device:", device)
    model = Gemma3WithInfiniAttention(CONFIG.beta, CONFIG.segment_length)
    optimizer = AdamW(model.parameters(), lr=CONFIG.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, CONFIG.warmup_step, CONFIG.train_step * CONFIG.epoch
    )
    print(model)

    start_epoch, losses = load_checkpoint(device, model, optimizer, scheduler)
    # export_loss(losses)

    # inspect_beta(model)

    tokenizer = AutoTokenizer.from_pretrained(CONST.model_name)
    # prompt = getTestPrompt(680, 680)
    prompt = getRandomPrompt()
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    print(tokens["input_ids"].shape)

    inspect_memory(model, tokens["input_ids"], tokens["attention_mask"])


if __name__ == "__main__":
    main()
