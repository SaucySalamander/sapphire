#!/usr/bin/env python3
"""Unified log comparator

Supports dynamic comparisons between two logs. Detects available data
and runs the appropriate comparisons:
- ATTN_VEC Q/K vector comparisons (per-layer, per-head RMSE/max_abs)
- Embedding & per-layer RMS comparisons
- Lightweight ATTENTION TRACE checks when present

Usage: python scripts/compare_logs.py fileA fileB
"""
import re
import sys
import math
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_attn_vecs(path: str) -> Dict[Tuple[int,int,int,str], List[float]]:
    """Parse lines like:
    ATTN_VEC: L4 H0 token=0 Q_vec: 0.123 0.234 ...
    Returns dict keyed by (layer, head, token, kind) where kind is 'Q' or 'K'
    """
    ATTN_RE = re.compile(r"^ATTN_VEC:\s+L(?P<layer>\d+)\s+H(?P<head>\d+)\s+token=(?P<token>\d+)\s+(?P<kind>[QK])_vec:\s*(?P<vals>.*)$")
    out = {}

    # Read full file so we can capture wrapped numeric lines that follow an ATTN_VEC header
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # robust float regex (floats and ints, with optional exponent)
    float_re = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?')

    i = 0
    while i < len(lines):
        line = lines[i]
        m = ATTN_RE.match(line.strip())
        if not m:
            i += 1
            continue
        layer = int(m.group('layer'))
        head = int(m.group('head'))
        token = int(m.group('token'))
        kind = m.group('kind')
        vals_s = m.group('vals').strip()

        # Collect following wrapped lines that contain only numeric tokens (and whitespace)
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if nxt == '':
                break
            if re.fullmatch(r'[-+0-9\.eE\s]+', nxt):
                vals_s += ' ' + nxt
                j += 1
                continue
            break

        vals = [float(x) for x in float_re.findall(vals_s)] if vals_s else []
        out[(layer, head, token, kind)] = vals

        i = j

    return out


def rmse(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float('nan')
    s = 0.0
    for i in range(n):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s / n)


def max_abs(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float('nan')
    return max(abs(a[i] - b[i]) for i in range(n))


def compare_attn_vecs(fileA: str, fileB: str) -> None:
    a = parse_attn_vecs(fileA)
    b = parse_attn_vecs(fileB)
    keys = sorted(set(a.keys()) & set(b.keys()), key=lambda k: (k[0], k[1], k[2], k[3]))
    if not keys:
        print('No overlapping ATTN_VEC entries found.')
        return
    print('layer head token kind n rmse max_abs')
    worst = (float('-inf'), None)
    for k in keys:
        va = a[k]
        vb = b[k]
        n = min(len(va), len(vb))
        r = rmse(va, vb)
        m = max_abs(va, vb)
        print(f"{k[0]:5d} {k[1]:4d} {k[2]:5d} {k[3]:4s} {n:3d} {r:12.6g} {m:12.6g}")
        if math.isfinite(r) and r > worst[0]:
            worst = (r, k)
    wr, wk = worst
    if wk:
        print(f"\nWORST: L{wk[0]} H{wk[1]} token={wk[2]} {wk[3]}_vec rmse={wr}")


def parse_torch_rms(text: str):
    # Embedding RMS
    emb = None
    m = re.search(r"Embedding RMS:\s*([0-9.eE+-]+)", text)
    if m:
        emb = float(m.group(1))
    # layer Output RMS lines
    layers = re.findall(r"Layer\s*(\d+) Output RMS:\s*([0-9.eE+-]+)", text)
    layers = sorted((int(i), float(v)) for i, v in layers)
    per_layer = [v for i, v in layers]
    return emb, per_layer


def parse_sapphire_rms(text: str):
    emb = None
    m = re.search(r"DEBUG\[EMBED\]: .*?after_scale rms=([0-9.eE+-]+)", text)
    if m:
        emb = float(m.group(1))
    # find post-ffn norm_buf RMS per layer
    found = {}
    for m in re.finditer(r"Layer\s*(\d+).*?post-ffn norm_buf RMS=([0-9.eE+-]+)", text):
        i = int(m.group(1)); v = float(m.group(2))
        if i not in found:
            found[i] = v
    if not found:
        for m in re.finditer(r"Layer\s*(\d+).*?post-attn norm_buf RMS=([0-9.eE+-]+)", text):
            i = int(m.group(1)); v = float(m.group(2))
            if i not in found:
                found[i] = v
    per_layer = [found[i] for i in sorted(found.keys())]
    return emb, per_layer


def compare_rms(fileA: str, fileB: str) -> None:
    with open(fileA, 'r', encoding='utf-8', errors='replace') as f:
        a_text = f.read()
    with open(fileB, 'r', encoding='utf-8', errors='replace') as f:
        b_text = f.read()

    a_torch = bool(re.search(r"Embedding RMS:", a_text))
    b_torch = bool(re.search(r"Embedding RMS:", b_text))

    if a_torch and b_torch:
        a_emb, a_layers = parse_torch_rms(a_text)
        b_emb, b_layers = parse_torch_rms(b_text)
    else:
        a_emb, a_layers = parse_sapphire_rms(a_text)
        b_emb, b_layers = parse_sapphire_rms(b_text)

    print('Embedding RMS — A: {:.4f}  B: {:.4f}'.format(a_emb or 0.0, b_emb or 0.0))
    n = max(len(a_layers), len(b_layers))
    print('\nPer-layer comparison:')
    print('Layer | A RMS | B RMS | Diff')
    print('----- | -----:| -----:| ----:')
    for i in range(n):
        ar = a_layers[i] if i < len(a_layers) else float('nan')
        br = b_layers[i] if i < len(b_layers) else float('nan')
        diff = (ar - br) if (not math.isnan(ar) and not math.isnan(br)) else float('nan')
        print(f'{i:5d} | {ar:7.4f} | {br:7.4f} | {diff:7.4f}')


def detect_presence(path: str):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        s = f.read()
    has_attn_vec = bool(re.search(r'^ATTN_VEC:', s, re.M))
    has_embedding_rms = bool(re.search(r'Embedding RMS:|DEBUG\[EMBED\]:', s))
    return {'attn_vec': has_attn_vec, 'rms': has_embedding_rms}


def main():
    if len(sys.argv) != 3:
        print('usage: scripts/compare_logs.py fileA fileB')
        return 2
    a, b = sys.argv[1], sys.argv[2]

    a_meta = detect_presence(a)
    b_meta = detect_presence(b)

    # If both files have ATTN_VEC, run vector compare
    if a_meta['attn_vec'] and b_meta['attn_vec']:
        print('\n== ATTN_VEC Comparison ==')
        compare_attn_vecs(a, b)
    else:
        print('\n== Skipping ATTN_VEC (not present in both files) ==')

    # If both files have RMS-ish data, run RMS compare
    if a_meta['rms'] and b_meta['rms']:
        print('\n== RMS Comparison ==')
        compare_rms(a, b)
    else:
        print('\n== Skipping RMS comparison (no embedding/RMS info in both files) ==')

    print('\n== Done ==')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
