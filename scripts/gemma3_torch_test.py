import math

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

print(f"CONFIG: heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim} query_pre_attn_scalar={q_pre} head_scalar={head_scalar}")

# IMPORTANT: Use the exact token sequence Sapphire uses for prompt "Hello".
# From Sapphire logs:
#   [2, 105, 2364, 107, 0, 9259, 106, 105, 4368, 107]
input_ids = torch.tensor([[2, 105, 2364, 107, 0, 9259, 106, 105, 4368, 107]], dtype=torch.long)
seq_len = int(input_ids.shape[1])
position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True, use_cache=False)

hidden_states = outputs.hidden_states
assert hidden_states is not None

dump_layers = [4, 13, 14, 15, 17]
token_pos = 0

for layer_idx in dump_layers:
    layer = text_model.layers[layer_idx]

    # Input to layer L is hidden_states[L] (embedding output is hidden_states[0]).
    x_in = hidden_states[layer_idx]  # [B, S, D]

    # Match Gemma3DecoderLayer: input_layernorm before attention.
    x_norm = layer.input_layernorm(x_in)

    # Projections
    q = layer.self_attn.q_proj(x_norm)
    k = layer.self_attn.k_proj(x_norm)

    # Reshape into heads
    q = q.view(1, seq_len, num_heads, head_dim)
    k = k.view(1, seq_len, num_kv_heads, head_dim)

    # QK-norm (per-head RMSNorm over head_dim)
    q = layer.self_attn.q_norm(q)
    k = layer.self_attn.k_norm(k)

    # RoPE (global vs local)
    rotary = _pick_rotary_emb(text_model, layer_idx)
    q, k = _apply_rope(q, k, rotary, position_ids)

    # Dump vectors for the requested token position.
    # K is shared across Q heads when num_kv_heads=1; Sapphire prints K_vec per head anyway.
    for h in range(num_heads):
        q_vec = q[0, token_pos, h, :]
        k_vec = k[0, token_pos, (h % num_kv_heads), :]
        print(f"ATTN_VEC: L{layer_idx} H{h} token={token_pos} Q_vec: {_fmt_vec(q_vec)}")
        print(f"ATTN_VEC: L{layer_idx} H{h} token={token_pos} K_vec: {_fmt_vec(k_vec)}")

    # Optional quick RMS sanity (mirrors what we often log in Sapphire).
    out_h = hidden_states[layer_idx + 1][0, token_pos, :]
    print(f"RMS: layer={layer_idx} token={token_pos} in={_rms(x_in[0, token_pos, :]):.6f} out={_rms(out_h):.6f}")