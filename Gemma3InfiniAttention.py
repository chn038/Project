import math
import torch
from transformers import AutoModelForCausalLM


class Activation(torch.nn.Module):
    def __init__(self, alpha=1.0, inplace=False):
        super(Activation, self).__init__()
        self.elu = torch.nn.ELU(alpha, inplace)

    def forward(self, x):
        return torch.add(self.elu(x), 1)


class Memory(torch.nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self.hidden_memory = None
        self.normalize_term = None

    def getMemory(self):
        return (self.hidden_memory, self.normalize_term)

    def clearMemory(self):
        self.hidden_memory = None
        self.normalize_term = None

    def updateMemory(self, hidden_memory, normalize_term):
        self.pending_memory = hidden_memory
        self.pending_norm = normalize_term


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
        hid_storage,
    ):
        super(Gemma3CompressiveMemory, self).__init__()
        self.dim_input = dim_input
        self.num_heads = num_heads
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.proj_q = torch.nn.Linear(
            dim_input, dim_key * num_heads, bias=False, dtype=torch.bfloat16
        )
        self.proj_k = torch.nn.Linear(
            dim_input, dim_key, bias=False, dtype=torch.bfloat16
        )
        self.proj_v = torch.nn.Linear(
            dim_input, dim_value, bias=False, dtype=torch.bfloat16
        )
        self.proj_out = torch.nn.Linear(
            dim_value * num_heads, dim_hidden, bias=False, dtype=torch.bfloat16
        )
        self.beta = beta
        self.q_norm = torch.nn.RMSNorm(dim_key, eps=eps, dtype=torch.bfloat16)
        self.k_norm = torch.nn.RMSNorm(dim_key, eps=eps, dtype=torch.bfloat16)
        self.act = Activation()
        self.softMax = torch.nn.Softmax(dim=3)
        self.hid_storage: Memory = hid_storage

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings=None,
        position_ids=None,
        past_key_values=None,
    ):
        """
        hidden_states Must be the shape of (batch_size, input_length, dim_input)
        attention_mask sent to the layer is already applied causal mask
            with shape (batch_size, 1, input_length, input_length)

        The a_dot will apply both attention mask and causal mask, so it will use (QK)V.
        However the memory will only apply attention mask, and since it's forced to use Q(KV),
        this will happened only in the update part.
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        hid, z = self.hid_storage.getMemory()

        if attention_mask is not None:
            causal_mask = torch.tril(
                torch.ones((seq_len, seq_len), device=device)
            ).bool()
            attn_mask_for_mem = attention_mask == causal_mask
            attn_mask_for_mem = torch.all(attn_mask_for_mem, dim=-2).unsqueeze(-1)
            attn_mask_for_cur = attention_mask
            mask_for_cur = attn_mask_for_cur & causal_mask

        if hid is None:
            hid = torch.zeros(
                (
                    batch_size,
                    1,
                    self.dim_key,
                    self.dim_value,
                ),
                dtype=dtype,
            ).to(device)

        if z is None:
            z = (
                torch.zeros((batch_size, 1, self.dim_key), dtype=dtype).to(device)
                + 1e-6
            )

        q = self.proj_q(hidden_states).view(batch_size, -1, seq_len, self.dim_key)
        k = self.proj_k(hidden_states).view(batch_size, -1, seq_len, self.dim_key)
        v = self.proj_v(hidden_states).view(batch_size, -1, seq_len, self.dim_value)

        q_norm = self.q_norm(q)
        k_norm = self.k_norm(k)
        q_act = self.act(q_norm)
        k_act = self.act(k_norm)

        # update hidden memory
        if attention_mask is not None:
            k_act_masked = k_act * attn_mask_for_mem
            v_masked = v * attn_mask_for_mem
        else:
            # if no mask, just don't apply anything
            k_act_masked = k_act
            v_masked = v

        v_diff = v_masked - torch.einsum(
            "bhsk, bhkv -> bhsv", k_act_masked, torch.div(hid, z)
        )
        hid_new = hid + torch.einsum("bhsk, bhsv -> bhkv", k_act_masked, v_diff)
        z_new = z + torch.sum(k_act_masked, dim=2)

        self.hid_storage.updateMemory(hid_new, z_new)

        # Do positional embeddings after updating memory
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            q_embed = (q_act * cos) + (self._rotate_half(q_act) * sin)
            k_embed = (k_act * cos) + (self._rotate_half(k_act) * sin)
        else:
            q_embed = q_act
            k_embed = k_act

        # calculate current attention

        # i == j == sseq_len, this is needed for Einstein notation
        attn_matrix = torch.einsum(
            "bhik, bhjk -> bhij", q_embed, k_embed / math.sqrt(self.dim_key)
        )
        if attention_mask is not None:
            attn_matrix = attn_matrix.masked_fill(~mask_for_cur, -1e9)

        a_dot_unflatten = torch.einsum(
            "bhss, bhsv -> bhsv", self.softMax(attn_matrix), v
        )

        a_dot_unflatten = torch.transpose(a_dot_unflatten, 1, 2)
        a_dot = a_dot_unflatten.reshape(
            (batch_size, seq_len, self.num_heads * self.dim_value)
        )

        # calculate attention from memory
        a_mem_unflatten = torch.einsum(
            "bhsk, bhkv -> bhsv", q_act, torch.div(hid, z.unsqueeze(-1))
        )
        a_mem_unflatten = torch.transpose(a_mem_unflatten, 1, 2)
        a_mem = a_mem_unflatten.reshape(
            (batch_size, seq_len, self.num_heads * self.dim_value)
        )

        # get attention
        a = self.beta * a_mem + (1 - self.beta) * a_dot

        out = self.proj_out(a)

        # The return value should be:
        # attention_out, attention_weight, kv cache
        # But since I'm not writing the regular attention and have a dedicated memory system
        # I'll just ignore those outputs
        return out, None


class Gemma3WithInfiniAttention(torch.nn.Module):
    def __init__(self, beta, segment_length=512):

        super(Gemma3WithInfiniAttention, self).__init__()

        self.original_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-270m-it"
        ).model
        self.segment_length = segment_length
        self.beta = beta

        # Extract model configuration from original model
        self._extract_model_config()

        # Create memory instances for each layer
        self.layer_memories = [Memory() for _ in range(self.num_layers)]

        # Replace attention layers in-place (no decoder modification needed)
        self._replace_attention_layers()

    @property
    def device(self):
        """Return the device of the model parameters."""
        return next(self.parameters()).device

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

        for i, layer in enumerate(self.original_model.layers):
            # Store original attention for weight copying
            original_attn = layer.self_attn

            # Select beta for this layer

            # Create new Infini-attention layer
            infini_attn = Gemma3CompressiveMemory(
                dim_input=self.dim_input,
                dim_key=self.dim_key,
                dim_value=self.dim_value,
                dim_hidden=self.dim_hidden,
                num_heads=self.num_heads,
                beta=self.beta,
                eps=self.eps,
                hid_storage=self.layer_memories[i],
            )

            # CRITICAL: Copy weights from original attention to Infini-attention
            self._copy_attention_weights_to_infini(original_attn, infini_attn)

            # Replace the attention layer
            layer.self_attn = infini_attn

    def _clear_all_memories(self):
        """Clear all layer memories before processing new sequence"""
        for memory in self.layer_memories:
            memory.clearMemory()

    def _get_next_token(self, output, temperature, top_k, top_p, do_sample):
        next_token_logits = output[0][:, -1, :] / temperature
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
        self._clear_all_memories()
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
        for _ in range(max_length):
            output = self(input_ids=input_ids, attention_mask=attention_mask)
            next_token = self._get_next_token(
                output, temperature, top_k, top_p, do_sample
            )
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1)).bool().to(self.device)], dim=1
            )

            if pad_token_id is not None and next_token == pad_token_id:
                break

        return input_ids

    # def generate(
    #     self,
    #     input_ids,
    #     attention_mask=None,
    #     max_length=50,
    #     temperature=1.0,
    #     do_sample=False,
    #     top_k=None,
    #     top_p=None,
    #     pad_token_id=None,
    #     **kwargs,
    # ):
    #     """
    #     Generate method with memory management
    #     """
    #     dtype = input_ids.dtype
    #     device = input_ids.device
    #     # Clear memories before generation
    #     self._clear_all_memories()
    #     self._switch_memory_mode("Training")
    #
    #     segments = self._segment_input(input_ids, attention_mask)
    #
    #     # only want the output from the last segment
    #     for segment_input_ids, segment_attention_mask in segments:
    #         output = self.original_model(
    #             input_ids=segment_input_ids, attention_mask=segment_attention_mask
    #         )
    #
    #     self._switch_memory_mode("Generating")
    #
    #     next_token = self._get_next_token(output, temperature, top_k, top_p, do_sample)
    #
    #     generated = input_ids.clone()
    #     generated = torch.cat([generated, next_token], dim=1)
    #     token_buffer = torch.empty((1, 0), dtype=dtype, device=device)
    #     token_buffer = torch.cat([token_buffer, next_token], dim=1)
    #
    #     current_pos = input_ids.shape[1]  # After initial prompt
    #     # we already have the first output
    #     for steps in range(max_length - 1):
    #         # Create appropriate attention mask for the buffer
    #         if attention_mask is not None:
    #             # Extend attention mask or create new one for current buffer
    #             segment_attn_mask = torch.ones(
    #                 (1, 1, token_buffer.shape[1], token_buffer.shape[1]),
    #                 device=device,
    #                 dtype=torch.bool,
    #             )
    #         else:
    #             segment_attn_mask = None
    #         position_ids = torch.arange(
    #             current_pos, current_pos + token_buffer.shape[1], device=device
    #         ).unsqueeze(0)
    #         output = self.original_model(
    #             token_buffer,
    #             position_ids=position_ids,
    #             attention_mask=segment_attn_mask,
    #         )
    #         next_token = self._get_next_token(
    #             output, temperature, top_k, top_p, do_sample
    #         )
    #
    #         if token_buffer.shape[1] == self.segment_length:
    #             # the extra token is the next start
    #             # update the memory
    #             self._manual_update_memory()
    #             # update pposition_ids
    #             current_pos += token_buffer.shape[1]
    #
    #             token_buffer = torch.empty((1, 0), dtype=dtype, device=device)
    #
    #         token_buffer = torch.cat([token_buffer, next_token], dim=1)
    #         generated = torch.cat([generated, next_token], dim=1)
    #
    #         if next_token == pad_token_id:
    #             break
    #
    #     return generated
