import math
import torch
from transformers import AutoModelForCausalLM, Gemma3TextConfig
from torch.utils.checkpoint import checkpoint
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)


class Activation(torch.nn.Module):
    def __init__(self, alpha=1.0, inplace=False):
        super(Activation, self).__init__()
        self.elu = torch.nn.ELU(alpha, inplace)

    def forward(self, x):
        return torch.add(self.elu(x), 1)


class Memory(torch.nn.Module):
    def __init__(self, ema_ratio=0.9):
        super(Memory, self).__init__()
        self.hidden_memory = None
        self.normalize_term = None
        self.pending_hidden_memory = None
        self.pending_normalize_term = None

    def getMemory(self):
        # return (None, None)
        return (self.hidden_memory, self.normalize_term)

    def clearMemory(self):
        self.hidden_memory = None
        self.normalize_term = None

    def updateMemory(self, hidden_memory, normalize_term):
        self.pending_hidden_memory = hidden_memory
        self.pending_normalize_term = normalize_term

        if self.training:
            self.flushMemory()

    def flushMemory(self):
        self.hidden_memory = self.pending_hidden_memory
        self.normalize_term = self.pending_normalize_term


class Gemma3CompressiveMemory(torch.nn.Module):
    def __init__(
        self,
        dim_input,
        dim_key,
        dim_value,
        dim_hidden,
        num_heads,
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
        self.q_norm = torch.nn.RMSNorm(dim_key, eps=eps, dtype=torch.bfloat16)
        self.k_norm = torch.nn.RMSNorm(dim_key, eps=eps, dtype=torch.bfloat16)
        self.act = Activation()
        self.softMax = torch.nn.Softmax(dim=3)
        self.hid_storage: Memory = hid_storage
        self.beta = torch.nn.Parameter(torch.zeros(num_heads))

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
        return checkpoint(
            self._infini_attention,
            hidden_states,
            attention_mask,
            position_embeddings,
            position_ids,
            past_key_values,
            use_reentrant=False,
        )

    def _infini_attention(
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
            attn_mask_for_mem = attention_mask & causal_mask
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
            z = torch.ones((batch_size, 1, self.dim_key), dtype=dtype).to(device)

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

        v_delta_nominator = torch.einsum("bhsk, bhkv -> bhsv", k_act_masked, hid)
        v_delta_denominator = torch.einsum("bhsk, bhk -> bhs", k_act_masked, z)
        v_delta = v_delta_nominator / v_delta_denominator.clamp(min=1e-6).unsqueeze(-1)

        v_diff = v_masked - v_delta
        hid_diff = torch.einsum("bhsk, bhsv -> bhkv", k_act_masked, v_diff)
        z_diff = torch.sum(k_act_masked, dim=2)

        hid_new = hid + hid_diff
        z_new = z + z_diff

        self.hid_storage.updateMemory(hid_new.detach(), z_new.detach())

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

        # calculate attention from memory
        a_mem_nominator = torch.einsum("bhsk, bhkv -> bhsv", q_act, hid)
        a_mem_denominator = torch.einsum("bhsk, bhk -> bhs", q_act, z)
        a_mem_unflatten = a_mem_nominator / a_mem_denominator.clamp(min=1e-6).unsqueeze(
            -1
        )
        a_mem_unflatten = torch.transpose(a_mem_unflatten, 1, 2)

        # get attention
        gate = torch.sigmoid(self.beta)
        gate = gate.view((1, 1, self.num_heads, 1))
        a_unflatten = gate * a_mem_unflatten + (1 - gate) * a_dot_unflatten
        a = a_unflatten.reshape(batch_size, seq_len, self.num_heads * self.dim_value)

        out = self.proj_out(a)
        # The return value should be:
        # attention_out, attention_weight, kv cache
        # But since I'm not writing the regular attention and have a dedicated memory system
        # I'll just ignore those outputs
        return out, None


class Gemma3WithInfiniAttention(torch.nn.Module):
    def __init__(self, beta, segment_length=512):

        super(Gemma3WithInfiniAttention, self).__init__()

        config = Gemma3TextConfig.from_pretrained("google/gemma-3-270m-it")
        config.sliding_window = segment_length

        self.original_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-270m-it", dtype="auto", config=config
        )

        # To save memory
        self.original_model.lm_head = checkpoint_wrapper(self.original_model.lm_head)
        self.segment_length = segment_length
        self.lm_head_segment_length = segment_length
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

        for i, layer in enumerate(self.original_model.model.layers):
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

        # pad inputs
        pad_len = (seq_len + self.segment_length - 1) // self.segment_length
        pad_len = pad_len * self.segment_length
        len_diff = pad_len - seq_len
        input_ids = torch.nn.functional.pad(
            input_ids, (0, len_diff), mode="constant", value=0
        )
        if attention_mask is not None:
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, len_diff), mode='constant', value=0
            )

        # Segment long sequences
        segments = []
        for start_idx in range(0, pad_len, self.segment_length):
            end_idx = min(start_idx + self.segment_length, seq_len)
            segment_input_ids = input_ids[:, start_idx:end_idx]

            segment_attention_mask = None
            if attention_mask is not None:
                segment_attention_mask = attention_mask[:, start_idx:end_idx]

            segments.append((segment_input_ids, segment_attention_mask))

        return segments

    def _manual_update_memory(self):
        for mem in self.layer_memories:
            mem.flushMemory()

    @torch.compile(fullgraph=True, backend="inductor")
    def _compute_segment_loss(
        self,
        segment_input_ids,
        segment_attention_mask,
        target_segment,
        chunk_size,
        **kwargs,
    ):
        """
        target_segments = tensor(batch_size, seg_len, 1)[seg_count]

        The loss function is implemented as cross entropy loss
        L = -zW_i + M_global + log(S_global)

        Return loss as sum
        """
        # if hasattr(self, "original_model"):
        #     first_layer = self.original_model.model.layers[-1].self_attn
        #     if hasattr(first_layer, "hid_storage"):
        #         hid, z = first_layer.hid_storage.getMemory()
        #         if hid is not None and z is not None:
        #             print(f"\n=== SEGMENT DEBUG [STARTING] ===")
        #             print(
        #                 f"  Layer -1 hid max/min: {hid.abs().max().item():.2e} / {hid.min().item():.2e}"
        #             )
        #             print(
        #                 f"  Layer -1 z value range: [{z.min().item():.2e}, {z.max().item():.2e}]"
        #             )
        #             print(
        #                 f"  Layer -1 avg z per head: {(z.squeeze(1).mean(dim=-1) / z.shape[-1]).item():.2e}"
        #             )
        #             print(
        #                 f"  Layer -1 hid / z: {(hid / z).min().item():.2e}, {(hid / z).max().item():.2e}"
        #             )
        #
        #             # Check memory utilization ratio
        #             ratio_check = hid.abs().sum() / (z.abs().sum() + 1e-10)
        #             print(f"  Memory activation ratio: {ratio_check.item():.2e}")
        #
        #             if z.max() > 1e5:
        #                 print(
        #                     f"⚠️ WARNING: z exceeded safe threshold ({z.max().item():.2e})"
        #                 )
        #             elif z.max() > 1e7:
        #                 print(f"🔴 CRITICAL: z approaching overflow limits!")
        #
        #             print(f"=====================================\n")
        output_segment = self.original_model.model(
            input_ids=segment_input_ids,
            attention_mask=segment_attention_mask,
            **kwargs,
        )[0]

        lm_head_weight = self.original_model.lm_head.weight

        target_weights_row = lm_head_weight[target_segment]

        z_w_i = (output_segment * target_weights_row).sum(dim=-1, keepdim=True)

        vocab_size = lm_head_weight.shape[0]
        end_idx_f = min(chunk_size, vocab_size)
        current_weights = lm_head_weight[:end_idx_f]
        segment_logits = torch.nn.functional.linear(output_segment, current_weights)
        segment_logits_fp32 = segment_logits.contiguous().float()

        global_max = torch.amax(segment_logits_fp32, dim=-1, keepdim=True)
        shifted_segment_logits = segment_logits_fp32 - global_max
        shifted_segment_logits = torch.clamp(shifted_segment_logits, min=-100, max=100)
        global_sum = torch.sum(torch.exp(shifted_segment_logits), dim=-1, keepdim=True)

        for i in range(end_idx_f, vocab_size, chunk_size):
            end_idx = min(i + chunk_size, vocab_size)
            current_weights = lm_head_weight[i:end_idx]

            segment_logits = torch.nn.functional.linear(output_segment, current_weights)
            segment_logits_fp32 = segment_logits.contiguous().float()
            local_max = torch.amax(segment_logits_fp32, dim=-1, keepdim=True)
            shifted_segment_logits = segment_logits_fp32 - local_max
            shifted_segment_logits = torch.clamp(
                shifted_segment_logits, min=-100, max=100
            )
            local_sum = torch.sum(
                torch.exp(shifted_segment_logits), dim=-1, keepdim=True
            )

            next_global_max = torch.maximum(global_max, local_max)
            next_global_sum = global_sum * torch.exp(
                global_max - next_global_max
            ) + local_sum * torch.exp(local_max - next_global_max)

            global_max = next_global_max
            global_sum = next_global_sum

        # global_max.shape = [batch_size, segment_length, 1]
        # global_sum.shape = [batch_size, segment_length, 1]
        # z_w_i.shape = [batch_size, segment_length, 1]
        loss = -z_w_i + torch.log(global_sum + 1e-10) + global_max
        # we will calculate sum loss here since we have no information about sequence length
        loss = torch.sum(loss)
        # if torch.isnan(loss) or torch.isinf(loss):
        #     print(f"  output_segment max: {output_segment.max().item()}")
        #     print(f"  z_w_i max: {z_w_i.max().item()}")
        #     print(f"  segment_logits max: {segment_logits.max().item()}")
        #     print(f"  global_sum min: {global_sum.min().item()}")
        #     raise RuntimeError("Stopped to prevent NaN propagation")

        return loss

    def computeLossForTraining(
        self,
        input_ids,
        attention_mask,
        target,
        gradient_accumulation_step=1,
        chunk_size=None,
        **kwargs,
    ):
        """
        Compute loss within each segment, and split out lm_head to reduce memory usage.
        Thus, a full rewrite of forward logit is needed.

        Return loss as a float
        """
        total_loss = 0.0
        self._clear_all_memories()
        segments = self._segment_input(input_ids, attention_mask)
        target_segments = self._segment_input(target, None)

        if chunk_size is None:
            chunk_size = self.original_model.lm_head.weight.shape[0]

        for i in range(len(segments)):
            segment_input_ids, segment_attention_mask = segments[i]
            target_segment, _ = target_segments[i]
            loss = self._compute_segment_loss(
                segment_input_ids,
                segment_attention_mask,
                target_segment,
                chunk_size=chunk_size,
                **kwargs,
            )

            sequence_length = input_ids.shape[-1]
            batch_size = input_ids.shape[0]

            # compute mean loss
            loss = loss / (batch_size * sequence_length * gradient_accumulation_step)
            loss.backward()

            total_loss += loss.item()

            del segment_input_ids, segment_attention_mask, loss

        self._clear_all_memories()  # save memory
        return total_loss

    def computeLossForTesting(self, input_ids, attention_mask, target, chunk_size=None):
        total_loss = 0.0
        self._clear_all_memories()
        segments = self._segment_input(input_ids, attention_mask)
        target_segments = self._segment_input(target, None)

        if chunk_size is None:
            chunk_size = self.original_model.lm_head.weight.shape[0]

        for i in range(len(segments)):
            segment_input_ids, segment_attention_mask = segments[i]
            target_segment, _ = target_segments[i]
            loss = self._compute_segment_loss(
                segment_input_ids,
                segment_attention_mask,
                target_segment,
                chunk_size=chunk_size,
            )
            self._manual_update_memory()

            sequence_length = input_ids.shape[-1]
            batch_size = input_ids.shape[0]

            # compute mean loss
            loss = loss / (batch_size * sequence_length)
            total_loss += loss.item()

        self._clear_all_memories()  # save memory
        return total_loss

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
            segment_outputs.append(outputs.logits)  # Last hidden state

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
        # forcing the model to be in eval mode to make sure memory is correct
        self.eval()
        # Clear memories before generation
        self._clear_all_memories()

        segments = self._segment_input(input_ids, attention_mask)

        # only want the output from the last segment
        for segment_input_ids, segment_attention_mask in segments:
            output = self.original_model(
                input_ids=segment_input_ids, attention_mask=segment_attention_mask
            )
            self._manual_update_memory()

        next_token = self._get_next_token(output, temperature, top_k, top_p, do_sample)

        generated = input_ids.clone()
        generated = torch.cat([generated, next_token], dim=1)

        last_segment, last_segment_attn_mask = segments[-1]

        idx = (last_segment_attn_mask == 0).nonzero(as_tuple=True)
        if idx[0].numel() <= 0:
            last_segment = torch.zeros_like(last_segment)
            last_segment_attn_mask = torch.zeros_like(last_segment_attn_mask)
            idx = (last_segment_attn_mask == 0).nonzero(as_tuple=True)
            self._manual_update_memory()

        idx = tuple(i[0] for i in idx)
        last_segment[idx] = next_token
        last_segment_attn_mask[idx] = 1

        # we already have the first output
        for steps in range(max_length - 1):
            output = self.original_model(
                input_ids=last_segment,
                attention_mask=last_segment_attn_mask
            )
            next_token = self._get_next_token(
                output, temperature, top_k, top_p, do_sample
            )

            generated = torch.cat([generated, next_token], dim=1)

            if next_token == pad_token_id:
                break

            idx = (last_segment_attn_mask == 0).nonzero(as_tuple=True)
            if idx[0].numel() <= 0:
                last_segment = torch.zeros_like(last_segment)
                last_segment_attn_mask = torch.zeros_like(last_segment_attn_mask)
                idx = (last_segment_attn_mask == 0).nonzero(as_tuple=True)
                self._manual_update_memory()

            idx = tuple(i[0] for i in idx)
            last_segment[idx] = next_token
            last_segment_attn_mask[idx] = 1

        self._clear_all_memories()

        return generated
