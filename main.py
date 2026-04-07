import marimo

__generated_with = "0.22.4"
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
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it").to(device)
    print(f"Model Config: {model.config}")
    print(model)
    return model, tokenizer


@app.function
def getModelOutput(model, tokenizer, x, y, passkey: int = 9054):
    model.eval()
    prompt = getTestPrompt(x, y, passkey)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", tokenize=True, add_generation_prompt=True
    ).to(model.device)
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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


@app.class_definition
class Activation(torch.nn.Module):
    def __init__(self, alpha=1.0, inplace=False):
        super(Activation, self).__init__()
        self.elu = torch.nn.ELU(alpha, inplace)

    def forward(self, x):
        return torch.add(self.elu(x), 1)


@app.class_definition
class Memory:
    def __init__(self):
        self.hidden_memory = None
        self.normalize_term = None
        self.mode = "Training"
        self.pending_memory = None
        self.pending_norm = None

    def updateMemory(self):
        self.hidden_memory = self.pending_memory
        self.normalize_term = self.pending_norm

    def getMemory(self):
        return (self.hidden_memory, self.normalize_term)

    def clearMemory(self):
        self.hidden_memory = None
        self.normalize_term = None

    def registerMemory(self, hidden_memory, normalize_term):
        self.pending_memory = hidden_memory
        self.pending_norm = normalize_term
        if self.mode == "Training":
            self.updateMemory()

    def switchMode(self, mode: str):
        self.mode = mode


@app.class_definition
class Gemma3CompressiveMemory(torch.nn.Module):
    def __init__(
        self,
        dim_input,
        dim_key,
        dim_value,
        dim_hidden,
        num_heads,
        beta,
        eps,
        seq_len,
        hid_storage,
    ):
        super(Gemma3CompressiveMemory, self).__init__()
        self.dim_input = dim_input
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.proj_q = torch.nn.Linear(dim_input, dim_key * num_heads, bias=False)
        self.proj_k = torch.nn.Linear(dim_input, dim_key, bias=False)
        self.proj_v = torch.nn.Linear(dim_input, dim_value, bias=False)
        self.proj_out = torch.nn.Linear(dim_value * num_heads, dim_hidden, bias=False)
        self.beta = beta
        self.q_norm = torch.nn.RMSNorm(dim_key, eps=eps)
        self.k_norm = torch.nn.RMSNorm(dim_key, eps=eps)
        self.act = Activation()
        self.softMax = torch.nn.Softmax(dim=3)
        self.hid_storage: Memory = hid_storage

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        x Must be the shape of (batch_size, input_length, dim_input)
        hid Must be the shape of (batch_size, dim_key, dim_key)
        attn_mask Must be the shape of (batch_size, input_length)

        The a_dot will apply both attention mask and causal mask, so it will use (QK)V.
        However the memory will only apply attention mask, and since it's forced to use Q(KV),
        this will happened only in the update part.
        """
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]

        hid, z = self.hid_storage.getMemory()
        z = z.view(-1, 1, 1) + 1e-6

        if attn_mask is not None:
            attn_mask_for_mem = attn_mask.unsqueeze(-1)
            attn_mask_for_cur = attn_mask[:, None, None, :].bool()
            causal_mask = torch.tril(
                torch.ones((self.seq_len, self.seq_len), device=device)
            ).bool()
            mask_for_cur = attn_mask_for_cur & causal_mask

        if hid is None:
            hid = torch.zeros(
                (
                    batch_size,
                    self.dim_key,
                    self.dim_value,
                ),
                dtype=dtype,
            ).to(device)

        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q_chunks = q.view((batch_size, self.num_heads, self.seq_len, self.dim_key))
        q_norm = self.q_norm(q_chunks)
        k_norm = self.k_norm(k)
        q_act = self.act(q_norm)
        k_act = self.act(k_norm)

        # update hidden memory
        if attn_mask is not None:
            k_act_masked = k_act * attn_mask_for_mem
            v_masked = v * attn_mask_for_mem
        else:
            # if no mask, just don't apply anything
            k_act_masked = k_act
            v_masked = v

        v_diff = v_masked - torch.einsum(
            "bsk, bkv -> bsv", k_act_masked, torch.div(hid, z)
        )
        hid_new = hid + torch.einsum("bsk, bsv -> bkv", k_act_masked, v_diff)
        z_new = z + torch.sum(k_act_masked, dim=2)

        self.hid_storage.registerMemory(hid_new, z_new)

        # calculate current attention
        attn_matrix = torch.einsum(
            "bhsk, bsk -> bhss", q_chunks, k_act / math.sqrt(self.dim_key)
        )
        if attn_mask is not None:
            attn_matrix = attn_matrix.masked_fill(~mask_for_cur, -1e9)

        a_dot_unflatten = torch.einsum(
            "bhss, bsv -> bhsv", self.softMax(attn_matrix), v
        )

        a_dot_unflatten = torch.transpose(a_dot_unflatten, 1, 2)
        a_dot = a_dot_unflatten.reshape(
            (batch_size, self.seq_len, self.num_heads * self.dim_value)
        )

        # calculate attention from memory
        a_mem_unflatten = torch.einsum("bhsk, bkv -> bhsv", q_act, torch.div(hid, z))
        a_mem_unflatten = torch.transpose(a_mem_unflatten, 1, 2)
        a_mem = a_mem_unflatten.reshape(
            (batch_size, self.seq_len, self.num_heads * self.dim_value)
        )

        # get attention
        a = self.beta * a_mem + (1 - self.beta) * a_dot

        out = self.proj_out(a)
        return out


@app.class_definition
class Gemma3WithInfiniAttention:
    def __init__(self, original_model, segment_length=512):
        self.original_model = original_model
        self.segment_length = segment_length

        # Extract model configuration from original model
        self._extract_model_config()

        # Create memory instances for each layer
        self.layer_memories = [Memory() for _ in range(self.num_layers)]

        # Replace attention layers in-place (no decoder modification needed)
        self._replace_attention_layers()

    def _extract_model_config(self):
        """Extract configuration from the original Gemma3 model"""
        # Gemma3-270m-it specific dimensions from your structure
        self.dim_input = 640  # embedding size
        self.dim_key = 256  # k_proj/v_proj output size
        self.dim_value = 256  # same as key
        self.dim_hidden = 640  # o_proj output size
        self.num_heads = 4  # q_proj: 640→1024, 1024/256=4
        self.num_layers = 18  # from (0-17): 18 layers
        self.eps = 1e-06  # Gemma3RMSNorm epsilon

    def _copy_attention_weights_to_infini(self, original_attn, infini_attn):
        """
        Copy weights from original Gemma3Attention to Gemma3CompressiveMemory
        Since both have identical weight structures, this is a direct copy
        """
        # Copy query projection weights
        with torch.no_grad():
            infini_attn.proj_q.weight.copy_(original_attn.q_proj.weight)

        # Copy key projection weights
        with torch.no_grad():
            infini_attn.proj_k.weight.copy_(original_attn.k_proj.weight)

        # Copy value projection weights
        with torch.no_grad():
            infini_attn.proj_v.weight.copy_(original_attn.v_proj.weight)

        # Copy output projection weights
        with torch.no_grad():
            infini_attn.proj_out.weight.copy_(original_attn.o_proj.weight)

        # Copy normalization layer weights (if they have learnable parameters)
        # Gemma3RMSNorm has learnable weights
        if hasattr(original_attn, "q_norm") and hasattr(infini_attn, "q_norm"):
            with torch.no_grad():
                infini_attn.q_norm.weight.copy_(original_attn.q_norm.weight)

        if hasattr(original_attn, "k_norm") and hasattr(infini_attn, "k_norm"):
            with torch.no_grad():
                infini_attn.k_norm.weight.copy_(original_attn.k_norm.weight)

        # Optional: Copy bias terms if they exist (Gemma3 typically uses bias=False)
        # But including for completeness
        def _copy_bias_if_exists(src, dst):
            if (
                hasattr(src, "bias")
                and src.bias is not None
                and hasattr(dst, "bias")
                and dst.bias is not None
            ):
                with torch.no_grad():
                    dst.bias.copy_(src.bias)

        _copy_bias_if_exists(original_attn.q_proj, infini_attn.proj_q)
        _copy_bias_if_exists(original_attn.k_proj, infini_attn.proj_k)
        _copy_bias_if_exists(original_attn.v_proj, infini_attn.proj_v)
        _copy_bias_if_exists(original_attn.o_proj, infini_attn.proj_out)
        _copy_bias_if_exists(original_attn.q_norm, infini_attn.q_norm)
        _copy_bias_if_exists(original_attn.k_norm, infini_attn.k_norm)

    def _replace_attention_layers(self):
        """Replace each Gemma3Attention with Gemma3CompressiveMemory and copy weights"""
        beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for i, layer in enumerate(self.original_model.model.layers):
            # Store original attention for weight copying
            original_attn = layer.self_attn

            # Select beta for this layer
            beta = beta_values[i % len(beta_values)]

            # Create new Infini-attention layer
            infini_attn = Gemma3CompressiveMemory(
                dim_input=self.dim_input,
                dim_key=self.dim_key,
                dim_value=self.dim_value,
                dim_hidden=self.dim_hidden,
                num_heads=self.num_heads,
                beta=beta,
                eps=self.eps,
                seq_len=self.segment_length,
                hid_storage=self.layer_memories[i],
            )

            # CRITICAL: Copy weights from original attention to Infini-attention
            self._copy_attention_weights_to_infini(original_attn, infini_attn)

            # Replace the attention layer
            layer.self_attn = infini_attn

            # Optional: Keep reference for debugging or inspection
            layer._original_attn = original_attn
            layer._infini_attn = infini_attn

    def _clear_all_memories(self):
        """Clear all layer memories before processing new sequence"""
        for memory in self.layer_memories:
            memory.clearMemory()

    def _switch_memory_mode(self, mode):
        for memory in self.layer_memories:
            memory.switchMode(mode)

    def _manual_update_memory(self):
        for memory in self.layer_memories:
            memory.updateMemory()

    def _get_next_token(self, output, temperature, top_k, top_p, do_sample):
        next_token_logits = output.logits[:, -1, :] / temperature
        if do_sample:
            # Apply top-k then top-p filtering
            if top_k is not None:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        return next_token

    def _segment_input(self, input_ids, attention_mask=None):
        """Segment input into chunks for processing"""
        batch_size, seq_len = input_ids.shape

        # Handle short sequences
        if seq_len <= self.segment_length:
            return [(input_ids, attention_mask)]

        # Segment long sequences
        segments = []
        for start_idx in range(0, seq_len, self.segment_length):
            end_idx = min(start_idx + self.segment_length, seq_len)
            segment_input_ids = input_ids[:, start_idx:end_idx]

            segment_attention_mask = None
            if attention_mask is not None:
                segment_attention_mask = attention_mask[:, start_idx:end_idx]

            segments.append((segment_input_ids, segment_attention_mask))

        return segments

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass with:
        1. Memory clearing
        2. Input segmentation
        3. Segment-by-segment processing
        4. Output reassembly
        """
        # Clear memories for new sequence
        self._clear_all_memories()

        self._switch_memory_mode("Training")

        # Segment input
        segments = self._segment_input(input_ids, attention_mask)

        # Process each segment
        segment_outputs = []

        for segment_input_ids, segment_attention_mask in segments:
            # Forward pass through original model (with replaced attention layers)
            outputs = self.original_model(
                input_ids=segment_input_ids,
                attention_mask=segment_attention_mask,
                **kwargs,
            )
            segment_outputs.append(outputs[0])  # Last hidden state

        # Concatenate outputs: [B, total_seq_len, dim_hidden]
        if len(segment_outputs) == 1:
            final_hidden_states = segment_outputs[0]
        else:
            final_hidden_states = torch.cat(segment_outputs, dim=1)

        # Return in standard format matching original model output
        # Most HuggingFace models return tuple: (last_hidden_state, ...)
        return (final_hidden_states,) + outputs[1:]

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=50,
        temperature=1.0,
        do_sample=False,
        top_k=None,
        top_p=None,
        pad_token_id=None,
        **kwargs,
    ):
        """
        Generate method with memory management
        """
        dtype = input_ids.dtype
        device = input_ids.device
        # Clear memories before generation
        self._clear_all_memories()
        self._switch_memory_mode("Training")

        segments = self._segment_input(input_ids, attention_mask)

        # only want the output from the last segment
        for segment_input_ids, segment_attention_mask in segments:
            output = self.original_model(
                input_ids=segment_input_ids, attention_mask=segment_attention_mask
            )

        self._switch_memory_mode("Generating")

        next_token = self._get_next_token(output, temperature, top_k, top_p, do_sample)

        generated = input_ids.clone()
        generated = torch.cat([generated, next_token], dim=1)
        token_buffer = torch.empty((1, 0), dtype=dtype, device=device)
        token_buffer = torch.cat([token_buffer, next_token], dim=1)

        # we already have the first output
        for steps in range(max_length - 1):
            output = self.original_model(token_buffer)
            next_token = self._get_next_token(
                output, temperature, top_k, top_p, do_sample
            )

            if token_buffer.shape[1] == self.segment_length:
                # the extra token is the next start
                # update the memory
                self._manual_update_memory()
                token_buffer = torch.empty((1, 0), dtype=dtype, device=device)

            token_buffer = torch.cat([token_buffer, next_token], dim=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


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
