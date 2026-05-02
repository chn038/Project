import marimo

__generated_with = "0.23.1"
app = marimo.App()

with app.setup:
    import random
    import marimo as mo
    import torch
    from tqdm import tqdm
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        pipeline,
        LlamaConfig,
    )
    from matplotlib import pyplot as plt
    from Gemma3InfiniAttention import Gemma3WithInfiniAttention
    from CustomDataset import getDataPath, CustomDataset
    import pandas as pd

    #    model_name = "google/gemma-3-270m-it"
    ref_model_name = "meta-llama/Meta-Llama-3-8B"
    model_name = "nanotron/llama3-8b-infini-attention"

    device = "cuda" if torch.accelerator.is_available() else "cpu"
    print(device)


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
def _():
    _prompt = getTestPrompt(5, 5)
    print(_prompt)
    return


@app.cell
def _(re):
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

    return (getModelOutput,)


@app.cell
def _(getModelOutput):
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

    return (passkeyRetrievalTask,)


@app.cell
def _():
    pass
    return


@app.function
def training_step(model, tokenizer, dataloader, optimizer):
    pass


@app.cell(disabled=True)
def _(passkeyRetrievalTask):
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
    tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
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
    return context_sizes, correctnesses, tokenizer


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


@app.cell(disabled=True)
def _():
    infini_attn_model = Gemma3WithInfiniAttention(0.1, 512).to(device)
    print(infini_attn_model)
    return (infini_attn_model,)


@app.cell(disabled=True)
def _():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    print(f"Model Config: {model.config}")
    print(model)
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
