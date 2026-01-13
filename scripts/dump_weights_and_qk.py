#!/usr/bin/env python3
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-270m-it"
TARGET_LAYERS = [4, 13, 14, 15, 17]

def tensor_stats(t):
    t = t.float()
    v = t.view(-1)
    mn = v.min().item()
    mx = v.max().item()
    mean = v.mean().item()
    std = v.std(unbiased=False).item()
    return mn, mx, mean, std

def print_weight_stats(name, w, num_heads=None):
    print(f"\nWEIGHT STATS: {name} shape={tuple(w.shape)} dtype={w.dtype}")
    mn, mx, mean, std = tensor_stats(w)
    print(f"  global min={mn:.6f} max={mx:.6f} mean={mean:.6f} std={std:.6f}")
    if num_heads is not None:
        out_dim = w.shape[0]
        head_rows = out_dim // num_heads
        if head_rows * num_heads == out_dim:
            for h in range(num_heads):
                sub = w[h*head_rows:(h+1)*head_rows, :]
                mn, mx, mean, std = tensor_stats(sub)
                print(f"  head {h:2d} rows={head_rows} min={mn:.6f} max={mx:.6f} mean={mean:.6f} std={std:.6f}")

def split_heads(vec, num_heads):
    # vec: 1D tensor length = num_heads * head_dim
    total = vec.numel()
    if total % num_heads == 0:
        head_dim = total // num_heads
        return vec.view(num_heads, head_dim)
    else:
        return vec.unsqueeze(0)

def main():
    print(f"Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    input_text = "Hello"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # single forward to get hidden states
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    sd = model.state_dict()
    num_heads = getattr(model.config, 'num_attention_heads', None)

    for L in TARGET_LAYERS:
        # Find q_proj/k_proj keys for layer L
        q_key = None
        k_key = None
        for k in sd.keys():
            if 'q_proj.weight' in k and (f'.{L}.' in k or f'layer.{L}' in k or f'layers.{L}' in k):
                q_key = k
            if 'k_proj.weight' in k and (f'.{L}.' in k or f'layer.{L}' in k or f'layers.{L}' in k):
                k_key = k
        if q_key is None or k_key is None:
            print(f"Layer {L}: q_proj or k_proj key not found (q={q_key} k={k_key}), skipping")
            continue

        q_w = sd[q_key].to(torch.float32)
        k_w = sd[k_key].to(torch.float32)
        q_b = sd.get(q_key.replace('.weight', '.bias'), None)
        k_b = sd.get(k_key.replace('.weight', '.bias'), None)
        if q_b is not None:
            q_b = q_b.to(torch.float32)
        if k_b is not None:
            k_b = k_b.to(torch.float32)

        print_weight_stats(f"{q_key}", q_w, num_heads)
        print_weight_stats(f"{k_key}", k_w, num_heads)

        # Take the hidden state at this layer for the last token
        h_state = outputs.hidden_states[L]
        last_tok = h_state[0, -1, :].to(torch.float32)

        q_out = torch.matmul(last_tok, q_w.t())
        k_out = torch.matmul(last_tok, k_w.t())
        if q_b is not None:
            q_out = q_out + q_b
        if k_b is not None:
            k_out = k_out + k_b

        # split into heads and print elementwise vectors
        print(f"\nLAYER {L} Q/K head vectors (last token):")
        q_heads = split_heads(q_out, num_heads if num_heads is not None else 1)
        k_heads = split_heads(k_out, num_heads if num_heads is not None else 1)
        for h in range(min(q_heads.size(0), 16)):
            qvec = q_heads[h].cpu().numpy()
            kvec = k_heads[h % k_heads.size(0)].cpu().numpy()
            qstr = ' '.join([f"{x:.6f}" for x in qvec])
            kstr = ' '.join([f"{x:.6f}" for x in kvec])
            print(f"ATTN_VEC: L{L} H{h} Q_vec: {qstr}")
            print(f"ATTN_VEC: L{L} H{h} K_vec: {kstr}")

if __name__ == '__main__':
    main()
