import math
import argparse
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _fmt_vec(vec: torch.Tensor) -> str:
    # Match Sapphire's ATTN_VEC format: floats separated by spaces
    vec = vec.detach().to(torch.float32).flatten().cpu()
    return " ".join(f"{float(x):.6f}" for x in vec)


def _rms(x: torch.Tensor) -> float:
    x = x.detach().to(torch.float32)
    return float(torch.sqrt(torch.mean(x * x)).item())


def _pick_rotary_emb(text_model, layer_idx: int):
    # Best-effort match to HF Gemma3 behavior: if a config provides a sliding window and
    # a list/interval of sliding layers, use local rotary for those layers.
    cfg = getattr(text_model, "config", None)
    use_local = False

    sliding_window = getattr(cfg, "sliding_window", None) if cfg is not None else None
    if sliding_window is not None:
        sliding_layers = getattr(cfg, "sliding_layers", None)
        if isinstance(sliding_layers, (list, tuple, set)):
            use_local = layer_idx in sliding_layers
        elif isinstance(sliding_layers, int):
            # If an integer is provided, interpret as: last N layers use sliding.
            n_layers = getattr(cfg, "num_hidden_layers", None)
            if n_layers is not None and sliding_layers > 0:
                use_local = layer_idx >= (n_layers - sliding_layers)
        else:
            # Fallback: assume no layer-specific sliding if not specified.
            use_local = False

    if use_local and hasattr(text_model, "rotary_emb_local"):
        return text_model.rotary_emb_local
    return text_model.rotary_emb


def _apply_rope(q: torch.Tensor, k: torch.Tensor, rotary_emb, position_ids: torch.Tensor):
    # Try to use HF's helper if present; otherwise fail loudly.
    try:
        from transformers.models.gemma3 import modeling_gemma3 as mg3
    except Exception as e:
        raise RuntimeError(f"Unable to import transformers gemma3 helpers: {e}")

    if not hasattr(mg3, "apply_rotary_pos_emb"):
        raise RuntimeError("transformers.models.gemma3.modeling_gemma3.apply_rotary_pos_emb not found")

    # transformers Gemma3 apply_rotary_pos_emb expects q/k shaped like [B, H, S, D]
    # (S is at dim=2), while we keep [B, S, H, D] for easier indexing.
    q_hsd = q.permute(0, 2, 1, 3).contiguous()
    k_hsd = k.permute(0, 2, 1, 3).contiguous()

    # HF rotary_emb modules typically accept (x, position_ids) and return (cos, sin)
    cos, sin = rotary_emb(q_hsd, position_ids)
    q_hsd, k_hsd = mg3.apply_rotary_pos_emb(q_hsd, k_hsd, cos, sin, position_ids)

    q = q_hsd.permute(0, 2, 1, 3).contiguous()
    k = k_hsd.permute(0, 2, 1, 3).contiguous()
    return q, k


# Dump post-(QK-norm + RoPE) Q/K vectors in the same ATTN_VEC format Sapphire prints.

parser = argparse.ArgumentParser()
parser.add_argument('--append-rms', action='store_true', help='Print embedding + per-layer RMS summary at end')
parser.add_argument('--dump-attn', action='store_true', help='Also print ATTN_VEC Q/K dumps (very verbose)')
parser.add_argument('--two-runs', action='store_true', help='Run second generation pass (disabled by default)')
parser.add_argument('--max-new-tokens', type=int, default=359, help='Maximum new tokens to generate (default: 359)')
args = parser.parse_args()

model_id = "google/gemma-3-270m-it"
print(f"Loading model {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)

text_model = model.model
cfg = model.config

num_heads = int(getattr(cfg, "num_attention_heads"))
num_kv_heads = int(getattr(cfg, "num_key_value_heads", 1))
head_dim = int(getattr(cfg, "head_dim"))
q_pre = float(getattr(cfg, "query_pre_attn_scalar"))
head_scalar = 1.0 / math.sqrt(q_pre)

print(model)
print(f"CONFIG: heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim} query_pre_attn_scalar={q_pre} head_scalar={head_scalar}")

# Build input ids via the tokenizer instead of hard-coded token ids.
messages = [{"role": "user", "content": "Write a short poem about the sea."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)

hidden_states = outputs.hidden_states
assert hidden_states is not None

dump_layers = list(range(int(cfg.num_hidden_layers)))
token_pos = 0
seq_len = inputs["input_ids"].shape[1]
position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

# storage for RMS computation when requested
layers_vals = defaultdict(list)
all_vals = []

if args.dump_attn:
        for layer_idx in dump_layers:
            layer = text_model.layers[layer_idx]

            # Input to layer L is hidden_states[L] (embedding output is hidden_states[0]).
            x_in = hidden_states[layer_idx]  # [B, S, D]

            # Match Gemma3DecoderLayer: input_layernorm before attention.
            x_norm = layer.input_layernorm(x_in)

            # Projections (keep originals for RMS checks before reshaping into heads)
            q_proj = layer.self_attn.q_proj(x_norm)
            k_proj = layer.self_attn.k_proj(x_norm)
            v_proj = None
            if hasattr(layer.self_attn, "v_proj"):
                v_proj = layer.self_attn.v_proj(x_norm)

            # Print pre-attention RMS diagnostics
            print(f"L{layer_idx} token={token_pos} RMS_input={_rms(x_in[0, token_pos, :]):.6f} RMS_input_norm={_rms(x_norm[0, token_pos, :]):.6f}")
            print(f"L{layer_idx} token={token_pos} RMS_q_proj={_rms(q_proj[0, token_pos, :]):.6f} RMS_k_proj={_rms(k_proj[0, token_pos, :]):.6f}"
                  + (f" RMS_v_proj={_rms(v_proj[0, token_pos, :]):.6f}" if v_proj is not None else ""))

            # Reshape into heads
            q = q_proj.view(1, seq_len, num_heads, head_dim)
            k = k_proj.view(1, seq_len, num_kv_heads, head_dim)
            v = None
            if v_proj is not None:
                v = v_proj.view(1, seq_len, num_kv_heads, head_dim)

            # QK-norm (per-head RMSNorm over head_dim)
            q = layer.self_attn.q_norm(q)
            k = layer.self_attn.k_norm(k)

            # RoPE (global vs local)
            rotary = _pick_rotary_emb(text_model, layer_idx)
            q, k = _apply_rope(q, k, rotary, position_ids)

            # ATTENTION TRACE dumps: per-head raw dot and scaled values (like Sapphire)
            try:
                has_qk_norm = 1 if (hasattr(layer.self_attn, 'q_norm') and hasattr(layer.self_attn, 'k_norm')) else 0
                for h in range(num_heads):
                    kv_idx = h % num_kv_heads
                    # raw dot against key pos 0
                    q_vec = q[0, token_pos, h, :]
                    k0_vec = k[0, 0, kv_idx, :]
                    raw0 = float(torch.dot(q_vec, k0_vec).item())
                    scaled0 = raw0 * head_scalar
                    print(f"ATTN_TRACE: L{layer_idx} H{h} head_scalar={head_scalar:.6f} raw_dot[0]={raw0:.6f} scaled[0]={scaled0:.6f} has_qk_norm={has_qk_norm}")
                    # per-key-position dump: raw, K0 (first element of k vector), scaled
                    for tpos in range(seq_len):
                        k_vec_t = k[0, tpos, kv_idx, :]
                        raw_t = float(torch.dot(q_vec, k_vec_t).item())
                        scaled_t = raw_t * head_scalar
                        k0 = float(k_vec_t[0].item()) if k_vec_t.numel() > 0 else 0.0
                        print(f"ATTN_TRACE: L{layer_idx} H{h} t={tpos} raw={raw_t:.6f} K0={k0:.6f} scaled={scaled_t:.6f}")
            except Exception as e:
                print(f"ATTN_TRACE compute failed: {e}")

            # Attention score matrix (pre-softmax) for head 0
            try:
                q_seq = q[0, :, 0, :]  # [S, D]
                k_seq = k[0, :, 0, :]  # [S, D] (uses kv-head 0)
                attn_scores = torch.matmul(q_seq, k_seq.transpose(0, 1)) / math.sqrt(float(head_dim))
                a_min = float(attn_scores.min().item())
                a_max = float(attn_scores.max().item())
                a_mean = float(attn_scores.mean().item())
                print(f"L{layer_idx} token={token_pos} ATTNSCORES_head0 pre-softmax min={a_min:.6f} max={a_max:.6f} mean={a_mean:.6f}")
            except Exception as e:
                print(f"L{layer_idx} token={token_pos} ATTNSCORES compute failed: {e}")

            # Build attention output per head (so we can pass through o_proj)
            attn_output_all = torch.zeros((1, seq_len, num_heads, head_dim), dtype=q.dtype, device=q.device)
            try:
                # softmax per query position across key positions
                attn_probs = torch.softmax(attn_scores, dim=-1)
                for h in range(num_heads):
                    kv_idx = h % num_kv_heads
                    if v is not None:
                        v_seq = v[0, :, kv_idx, :]  # [S, D]
                        attn_output_all[0, :, h, :] = torch.matmul(attn_probs, v_seq)
                    else:
                        # If v_proj missing, leave zeros
                        pass
            except Exception as e:
                print(f"L{layer_idx} token={token_pos} ATTENTION output build failed: {e}")

            # Flatten heads and run through o_proj if available
            attn_out_flat = attn_output_all.reshape(1, seq_len, num_heads * head_dim)
            o_proj_fn = getattr(layer.self_attn, "o_proj", None) or getattr(layer.self_attn, "out_proj", None)
            if o_proj_fn is not None:
                try:
                    o_out = o_proj_fn(attn_out_flat)
                except Exception as e:
                    print(f"L{layer_idx} token={token_pos} o_proj apply failed: {e}")
                    o_out = attn_out_flat
            else:
                o_out = attn_out_flat

            # RMS of o_proj output (before residual add) and RMS after residual add
            try:
                rms_o = _rms(o_out[0, token_pos, :])
                post_resid = o_out + x_in
                rms_post = _rms(post_resid[0, token_pos, :])
                print(f"L{layer_idx} token={token_pos} RMS_o_proj={rms_o:.6f} RMS_post_resid={rms_post:.6f}")
            except Exception as e:
                print(f"L{layer_idx} token={token_pos} RMS o/proj/resid compute failed: {e}")

            # Dump vectors for the requested token position.
            # K is shared across Q heads when num_kv_heads=1; Sapphire prints K_vec per head anyway.
            for h in range(num_heads):
                q_vec = q[0, token_pos, h, :]
                k_vec = k[0, token_pos, (h % num_kv_heads), :]
                print(f"ATTN_VEC: L{layer_idx} H{h} token={token_pos} Q_vec: {_fmt_vec(q_vec)}")
                print(f"ATTN_VEC: L{layer_idx} H{h} token={token_pos} K_vec: {_fmt_vec(k_vec)}")
                if args.append_rms:
                    q_list = [float(x) for x in q_vec.detach().to(torch.float32).flatten().cpu().tolist()]
                    k_list = [float(x) for x in k_vec.detach().to(torch.float32).flatten().cpu().tolist()]
                    layers_vals[layer_idx].extend(q_list)
                    layers_vals[layer_idx].extend(k_list)
                    all_vals.extend(q_list)
                    all_vals.extend(k_list)

            # Optional quick RMS sanity (mirrors what we often log in Sapphire).
            out_h = hidden_states[layer_idx + 1][0, token_pos, :]
            print(f"RMS: layer={layer_idx} token={token_pos} in={_rms(x_in[0, token_pos, :]):.6f} out={_rms(out_h):.6f}")

# Prepare input ids and attention mask (tokenizer may return it) and ensure proper pad token
input_ids = inputs["input_ids"].to(next(model.parameters()).device)
attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
attention_mask = attention_mask.to(input_ids.device)
pad_token = getattr(tokenizer, "eos_token_id", None)


def run_autoregressive(input_ids, attention_mask, max_new_tokens=80, do_sample=False, temperature=1.0, tag="greedy"):
    device = input_ids.device
    cur_ids = input_ids.clone()
    cur_attn = attention_mask.clone()
    generated = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(cur_ids, attention_mask=cur_attn, output_hidden_states=True, use_cache=False)

        # logits for next token
        logits = out.logits[:, -1, :]

        if do_sample:
            probs = torch.softmax(logits / float(max(1e-8, temperature)), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        # materialize token id scalar
        tok_id = int(next_id[0, 0].item()) if next_id.dim() > 1 else int(next_id[0].item())
        generated.append(tok_id)

        # Print token-level info
        tok_text = tokenizer.convert_ids_to_tokens([tok_id])[0]
        print(f"GEN_STEP: step={step} tag={tag} token_id={tok_id} token={tok_text}")

        # hidden_states: list len = num_layers+1, each [B, S, D]
        hstates = out.hidden_states
        if hstates is not None:
            last_pos = cur_ids.shape[1] - 1
            # Embedding RMS at last position
            emb_rms = _rms(hstates[0][0, last_pos, :])
            print(f"Embedding RMS: {emb_rms:.6f}")
            num_layers = int(getattr(cfg, "num_hidden_layers"))
            # for li in range(num_layers):
            #     out_vec = hstates[li + 1][0, last_pos, :]
            #     print(f"Layer {li} Output RMS: {float(torch.sqrt(torch.mean(out_vec.detach().to(torch.float32) * out_vec.detach().to(torch.float32))).item()):.6f}")

        # append token and continue
        # next_id may be shape [1,1] or [1]; normalize to [1,1]
        if next_id.dim() == 1:
            next_id = next_id.unsqueeze(1)
        cur_ids = torch.cat([cur_ids, next_id.to(device)], dim=1)
        cur_attn = torch.cat([cur_attn, torch.ones((cur_attn.shape[0], 1), dtype=cur_attn.dtype, device=device)], dim=1)

        # stop if EOS
        if pad_token is not None and tok_id == pad_token:
            break

    # decode generated portion
    if len(generated) > 0:
        gen_text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    else:
        gen_text = ""

    return generated, gen_text


# Run greedy (deterministic) autoregressive generation with per-token per-layer RMS
# Use the same (larger) token budget as Sapphire and keep generation deterministic (non-creative)
greedy_ids, greedy_text = run_autoregressive(input_ids, attention_mask, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0, tag="greedy")
print("\n--- Generated response (greedy) ---")
print(greedy_text)

# Optionally run a second pass; disabled by default to avoid duplicate logs
if args.two_runs:
    sample_ids, sample_text = run_autoregressive(input_ids, attention_mask, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0, tag="sample")
    print("\n--- Generated response (sampled-as-deterministic, temp=0.0) ---")
    print(sample_text)