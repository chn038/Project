import marimo

__generated_with = "0.23.0"
app = marimo.App()

with app.setup:
    import re
    import gc
    import random
    import marimo as mo
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from matplotlib import pyplot as plt
    from Gemma3InfiniAttention import Gemma3WithInfiniAttention
    import math

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
    prompt = getTestPrompt(5, 5)
    print(prompt)
    return


@app.cell
def _():
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
    model = model.to(device)
    print(f"Model Config: {model.config}")
    print(model)
    return model, tokenizer


@app.function
def getModelOutput(model, tokenizer, x, y, passkey: int = 9054):
    model.eval()
    prompt = getTestPrompt(x, y, passkey)
    # messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    # inputs = tokenizer.apply_chat_template(
    #     messages, return_tensors="pt", tokenize=True, add_generation_prompt=True
    # ).to(model.device)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_outputs = model.generate(
            **inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id
        )
    decoded_outputs = tokenizer.decode(
        generated_outputs[0][inputs["input_ids"].shape[-1] :]
    )
    outputs = re.search(r"\d+", decoded_outputs)
    output = outputs.group() if outputs is not None else None

    del inputs, generated_outputs
    return output, decoded_outputs


@app.function
def passkeyRetrievalTask(model, tokenizer, x, y, key_length=4, test_times=100):
    correct = 0
    for _ in range(test_times):
        passkey = random.randint(pow(10, key_length - 1), pow(10, key_length) - 1)
        output, raw_output = getModelOutput(model, tokenizer, x, y, passkey)
        if output is not None and int(output) == passkey:
            correct += 1

    return correct / test_times


@app.cell
def _(tokenizer):
    infini_attn_model = Gemma3WithInfiniAttention(0.1, 512).to(device)
    print(infini_attn_model)
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


@app.cell(disabled=True)
def _(model, tokenizer):
    context_sizes = []
    correctnesses = []
    for i in tqdm(range(0, 1300, 10)):
        x = i // 2
        y = i - x
        _prompt = getTestPrompt(x, y, pow(10, 4 - 1))
        messages = [{"role": "user", "content": [{"type": "text", "text": _prompt}]}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", tokenize=True, add_generation_prompt=True
        )
        # inputs = tokenizer(prompt, return_tensors="pt")
        context_sizes.append(inputs.input_ids.shape[1])
        correctness = passkeyRetrievalTask(model, tokenizer, x, y)
        correctnesses.append(correctness)

        gc.collect()
        torch.cuda.empty_cache()

    plt.plot(context_sizes, correctnesses)
    return


if __name__ == "__main__":
    app.run()
