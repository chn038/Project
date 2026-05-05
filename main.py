import marimo

__generated_with = "0.23.1"
app = marimo.App()

with app.setup:
    import glob
    import os
    import gc
    import random
    import math
    import re
    from itertools import cycle
    import marimo as mo
    import torch
    from torch.nn.functional import cross_entropy
    from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
    from torch.utils.data.datapipes.iter import IterableWrapper
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        pipeline,
        LlamaConfig,
        get_linear_schedule_with_warmup,
    )
    from matplotlib import pyplot as plt
    from Gemma3InfiniAttention import Gemma3WithInfiniAttention
    from CustomDataset import getDataPath, CustomDataset
    import pandas as pd

    model_name = "google/gemma-3-270m-it"
    DATA_FILE_COUNT = 15
    DATA_PATH = "fineweb/sample/10BT/"
    PROCESSED_PATH = "fineweb/tokenized/"
    CHECKPOINT_PATH = "checkpoints/"

    device = "cuda" if torch.accelerator.is_available() else "cpu"
    print(device)
    torch.set_float32_matmul_precision('high')


@app.class_definition
class CONFIG:
    train_test_ratio = 0.9
    train_step = 1000
    test_steps = 10
    epoch = 30
    buffer_size = int(1e4)
    worker_count = 4
    batch_size = 1
    context_size = 16384
    beta = 0.5
    segment_length = 512
    lr = 5e-5
    warmup_step = 10000


@app.cell
def _():
    run_memory_test_btn = mo.ui.run_button(kind="neutral", label="Test memory")
    run_tokenize_btn = mo.ui.run_button(kind="neutral", label="Re-tokenize the dataset")
    run_train_btn = mo.ui.run_button(kind="neutral", label="Run model training")
    mo.hstack([run_memory_test_btn, run_tokenize_btn, run_train_btn])
    return run_memory_test_btn, run_tokenize_btn, run_train_btn


@app.function
def getTestPrompt(x: int, y: int, passkey: int = 9054) -> str:
    instruct = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there. "
    placeHolder = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    passkey_phrase = (
        f"The pass key is {passkey}. Remember it. {passkey} is the pass key. "
    )
    endPhase = "What is the pass key? The pass key is"
    prompt = instruct + placeHolder * x + passkey_phrase + placeHolder * y + endPhase
    return prompt


@app.cell
def Test_Prompt():
    _prompt = getTestPrompt(5, 5)
    print(_prompt)
    return


@app.function
def getModelOutput(model, tokenizer, x, y, model_pipeline=None, passkey: int = 9054):
    prompt = getTestPrompt(x, y, passkey)

    if model_pipeline is None:
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_outputs = model.generate(
                **inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id
            )
        decoded_outputs = tokenizer.decode(
            generated_outputs[0][inputs["input_ids"].shape[-1] :]
        )
        del inputs, generated_outputs
    else:
        decoded_outputs = model_pipeline(prompt)

    outputs = re.search(r"\d+", decoded_outputs)
    output = outputs.group() if outputs is not None else None

    return output, decoded_outputs


@app.function
def passkeyRetrievalTask(
    model, tokenizer, model_pipeline, x, y, key_length=4, test_times=100
):
    correct = 0
    for _ in range(test_times):
        passkey = random.randint(pow(10, key_length - 1), pow(10, key_length) - 1)
        output, raw_output = getModelOutput(
            model, tokenizer, x, y, model_pipeline, passkey
        )
        if output is not None and int(output) == passkey:
            correct += 1
    return


@app.cell
def _(run_train_btn):
    mo.stop(not run_train_btn.value, "Press train button to run")
    _data_blocks = list(range(DATA_FILE_COUNT))
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
    return test_dataloader, train_dataloader


@app.cell
def _(run_train_btn):
    mo.stop(not run_train_btn.value, "Press train button to run")
    model = Gemma3WithInfiniAttention(CONFIG.beta, CONFIG.segment_length)
    optimizer = AdamW(model.parameters(), lr=CONFIG.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, CONFIG.warmup_step, CONFIG.train_step * CONFIG.epoch
    )
    return model, optimizer, scheduler


@app.function
def training_step(model, dataloader, optimizer, scheduler, steps):
    model.train()
    losses = 0
    loss_steps = []
    for step in tqdm(range(steps)):
        tokens, attn_mask = next(dataloader)
        tokens = tokens.to(model.device)
        attn_mask = attn_mask.to(model.device)
        labels = tokens.clone()
        labels[:, :-1] = tokens[:, 1:]
        logits = model(input_ids=tokens, attention_mask=attn_mask)
        logits = logits[0]
        loss = cross_entropy(logits, labels)
        step_loss = torch.sum(loss)
        losses += step_loss.item()
        loss_steps.append(step_loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print(f"Training loss: {losses / steps}")
    return loss_steps


@app.function
def testing_step(model, dataloader, steps):
    model.eval()
    losses = 0
    for step in range(steps):
        tokens, attn_mask = next(dataloader)
        tokens = tokens.to(model.device)
        attn_mask = attn_mask.to(model.device)
        labels = tokens.clone()
        labels[:, :-1] = tokens[:, 1:]
        with torch.no_grad():
            logits = model(input_ids=tokens, attention_mask=attn_mask)
        logits = logits[0]
        loss = cross_entropy(logits, labels)
        loss = torch.sum(loss)
        losses += loss.item()

    print(f"Test loss: {losses / steps}")


@app.function
def load_checkpoint(model, optimizer, scheduler):
    # Find all checkpoint files
    list_of_files = glob.glob(CHECKPOINT_PATH + "checkpoint_epoch_*.pth")
    if list_of_files:
        # Get the latest checkpoint file
        latest_file = max(
            list_of_files, key=os.path.getctime
        )  # or use os.path.getmtime
        # latest_file = max(list_of_files, key=os.path.getmtime)  # Alternative

        # Load it
        checkpoint = torch.load(latest_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        losses = checkpoint["loss"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} using {latest_file}")
    else:
        losses = []
        start_epoch = 0
        print("No checkpoint found, starting from scratch")
    return start_epoch, losses


@app.cell
def _(
    model,
    optimizer,
    run_train_btn,
    scheduler,
    test_dataloader,
    train_dataloader,
):
    mo.stop(not run_train_btn.value, "Press train button to run")
    start_epoch, losses = load_checkpoint(model, optimizer, scheduler)
    epoch = start_epoch
    # _model = model.to(device)
    _model = model
    while epoch < CONFIG.epoch:
        print(f"Epoch {epoch + 1}/{CONFIG.epoch}")
        loss_steps = training_step(_model, train_dataloader, optimizer, scheduler, CONFIG.train_step)
        losses.extend(loss_steps)
        testing_step(_model, test_dataloader, CONFIG.test_steps)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                'loss': losses,
            },
            CHECKPOINT_PATH + f"checkpoint_epoch_{epoch}.pth",
        )
        epoch += 1
    return


@app.cell(disabled=True)
def _():
    """
    model_config = LlamaConfig.from_dict(
        {
            "model_type": "llama",
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "is_llama_config": True,
            "max_position_embeddings": 8192,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "pad_token_id": None,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "rope_theta": 500000.0,
            "rope_interleaved": False,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 128256,
        }
    )
    model_pipeline = pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained(model_name, config=model_config),
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    context_sizes = []
    correctnesses = []
    for i in tqdm(range(0, 1300, 10)):
        x = i // 2
        y = i - x
        _prompt = getTestPrompt(x, y, pow(10, 4 - 1))
        inputs = tokenizer(_prompt, return_tensors="pt")
        context_sizes.append(inputs.input_ids.shape[1])
        correctness = passkeyRetrievalTask(None, None, model_pipeline, x, y)
        correctnesses.append(correctness)
    """
    pass
    return


@app.cell(disabled=True)
def _(context_sizes, correctnesses):
    data = pd.DataFrame({"Context Size": context_sizes, "Correctness": correctnesses})
    data.to_csv("testresult.csv")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    # The following cell is disabled for either too costy or meant for testing.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The next cell is to tokenize the database and store it into PROCESSED_PATH
    It is affected by buffer size and context size, so if any of these are changed, it needs to be re-run.
    """)
    return


@app.cell
def _(run_tokenize_btn):
    mo.stop(not run_tokenize_btn.value, "Press tokenize button to run")
    def _getDataPath(num):
        if num >= DATA_FILE_COUNT or num < 0:
            raise ValueError(
                f"Number should be between 0 and {DATA_FILE_COUNT - 1}, get {num} instead"
            )

        return f"{DATA_PATH}{num:03d}_00000.parquet"

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _seg_num = 0
    for _num in range(DATA_FILE_COUNT):
        print(f"Handling chunk {_num + 1}/{DATA_FILE_COUNT}")
        _df = pd.read_parquet(_getDataPath(_num))["text"]
        _tokens = []
        _attn_masks = []
        for _text in tqdm(_df):
            _tok = _tokenizer(_text)
            _token = _tok["input_ids"]
            _attn_mask = _tok["attention_mask"]
            _token.append(_tokenizer.eos_token_id)
            _attn_mask.append(1)
            _tokens.extend(_token)
            _attn_masks.extend(_attn_mask)

        del _token, _attn_mask, _df

        _tokens = torch.tensor(_tokens)
        _attn_masks = torch.tensor(_attn_masks)

        _total_length = len(_tokens)
        _lines = math.floor(_total_length / CONFIG.context_size)
        _trim_lines = (
            math.floor(_lines / CONFIG.buffer_size)
            * CONFIG.buffer_size
            * CONFIG.context_size
        )
        _tokens_block = _tokens[:_trim_lines]
        _tokens_remainder = _tokens[_trim_lines:]
        del _tokens
        _attn_masks_block = _attn_masks[:_trim_lines]
        _attn_masks_remainder = _attn_masks[_trim_lines:]
        del _attn_masks
        _remainder_length = _tokens_remainder.shape[0]
        _trim_length = (
            math.floor(_remainder_length / CONFIG.context_size) * CONFIG.context_size
        )
        _tokens_remainder = _tokens_remainder[:_trim_length]
        _attn_masks_remainder = _attn_masks_remainder[:_trim_length]
        _tokens_block = _tokens_block.view(-1, CONFIG.buffer_size, CONFIG.context_size)
        _attn_masks_block = _attn_masks_block.view(
            -1, CONFIG.buffer_size, CONFIG.context_size
        )
        _seg = _tokens_block.shape[0]
        for _i in range(_seg):
            _df = pd.DataFrame(
                {
                    "tokens": _tokens_block[_i].tolist(),
                    "attention_mask": _attn_masks_block[_i].tolist(),
                }
            )
            _df.to_parquet(f"{PROCESSED_PATH}{_seg_num:03d}_00000.parquet")
            _seg_num += 1
            del _df
        del _tokens_block, _attn_masks_block

        _tokens_remainder = _tokens_remainder.view(-1, CONFIG.context_size)
        _attn_masks_remainder = _attn_masks_remainder.view(-1, CONFIG.context_size)
        _df = pd.DataFrame(
            {
                "tokens": _tokens_remainder.tolist(),
                "attention_mask": _attn_masks_remainder.tolist(),
            }
        )
        _df.to_parquet(f"{PROCESSED_PATH}{_seg_num:03d}_00000.parquet")
        _seg_num += 1
        del _df
        del _tokens_remainder, _attn_masks_remainder
    print(_seg_num)
    return


@app.cell(disabled=True)
def _():
    infini_attn_model = Gemma3WithInfiniAttention(0.1, 512).to(device)
    print(infini_attn_model)
    return (infini_attn_model,)


@app.cell
def _(run_memory_test_btn):
    mo.stop(not run_memory_test_btn.value, "Press memory test button to run")
    _model = Gemma3WithInfiniAttention(CONFIG.beta, CONFIG.segment_length)
    _model = _model.to(device)
    _model.compile()
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _prompt = getTestPrompt(350, 350)
    _model.train()
    _inputs = _tokenizer(_prompt, return_tensors="pt").to(_model.device)
    _out = _model(**_inputs)
    print(_inputs["input_ids"].shape)
    print(_model)
    del _model, _out, _inputs
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return


@app.cell(disabled=True)
def _(load_dataset):
    fw = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )
    print(fw)
    return


@app.cell(disabled=True)
def _(infini_attn_model, tokenizer):
    while True:
        _prompt = input()
        if _prompt == "/exit":
            break
        _inputs = tokenizer(_prompt, return_tensors="pt").to(infini_attn_model.device)
        with torch.no_grad():
            _generated_outputs = infini_attn_model.generate(
                **_inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id
            )
        _decoded_outputs = tokenizer.decode(_generated_outputs)
        print(_generated_outputs)
        print(_decoded_outputs)
    return


if __name__ == "__main__":
    app.run()
