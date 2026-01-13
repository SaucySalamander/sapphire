#!/usr/bin/env python3
import re
import sys

def parse_torch(path):
    with open(path, 'r') as f:
        s = f.read()
    # find first [STEP 0] block
    m = re.search(r"\[STEP 0\](.*?)(?:\n\[STEP |\Z)", s, re.S)
    if not m:
        raise SystemExit('STEP 0 block not found in torch log')
    block = m.group(1)
    emb = re.search(r"Embedding RMS:\s*([0-9.eE+-]+)", block)
    layers = re.findall(r"Layer\s*(\d+) Output RMS:\s*([0-9.eE+-]+)", block)
    layers = sorted((int(i), float(v)) for i,v in layers)
    return float(emb.group(1)) if emb else None, [v for i,v in layers]

def parse_sapphire(path):
    with open(path,'r') as f:
        s = f.read()
    # embedding after_scale
    emb = None
    m = re.search(r"DEBUG\[EMBED\]: .*after_scale rms=([0-9.eE+-]+)", s)
    if m:
        emb = float(m.group(1))
    # collect post-ffn norm_buf RMS per layer (first occurrence per layer)
    found = {}
    for m in re.finditer(r"Layer\s*(\d+).*?post-ffn norm_buf RMS=([0-9.eE+-]+)", s):
        i = int(m.group(1)); v = float(m.group(2))
        if i not in found:
            found[i] = v
    # fallback: post-attn values if post-ffn missing
    if not found:
        for m in re.finditer(r"Layer\s*(\d+).*?post-attn norm_buf RMS=([0-9.eE+-]+)", s):
            i = int(m.group(1)); v = float(m.group(2))
            if i not in found:
                found[i] = v
    layers = [found[i] for i in sorted(found.keys())]
    return emb, layers

def main():
    if len(sys.argv) != 3:
        print('usage: scripts/compare_rms.py sapphire.log torch.log')
        raise SystemExit(2)
    sapp_log, torch_log = sys.argv[1], sys.argv[2]
    s_emb, s_layers = parse_sapphire(sapp_log)
    t_emb, t_layers = parse_torch(torch_log)

    print('Embedding RMS — Torch: {:.4f}  Sapphire: {:.4f}'.format(t_emb or 0.0, s_emb or 0.0))
    n = max(len(t_layers), len(s_layers))
    print('\nPer-layer comparison (Torch_vs_Sapphire, step 0):')
    print('Layer | Torch RMS | Sapphire RMS | Diff')
    print('----- | ---------:| -----------:| -----:')
    for i in range(n):
        tr = t_layers[i] if i < len(t_layers) else float('nan')
        sr = s_layers[i] if i < len(s_layers) else float('nan')
        diff = tr - sr
        print(f'{i:5d} | {tr:9.4f} | {sr:11.4f} | {diff:7.4f}')

    # highlight top discrepancies
    pairs = []
    for i in range(min(len(t_layers), len(s_layers))):
        pairs.append((abs(t_layers[i]-s_layers[i]), i, t_layers[i], s_layers[i]))
    pairs.sort(reverse=True)
    print('\nTop 5 layer discrepancies (abs diff):')
    for d,i,tr,sr in pairs[:5]:
        print(f'Layer {i}: Torch={tr:.4f} Sapphire={sr:.4f} abs_diff={d:.4f}')

    # --- Layer-0 per-head QK comparison ---
    def parse_torch_attn(path):
        with open(path, 'r') as f:
            s = f.read()
        # split into STEP blocks
        steps = {}
        for m in re.finditer(r"\[STEP (\d+)\](.*?)(?=\n\[STEP |\Z)", s, re.S):
            step = int(m.group(1))
            block = m.group(2)
            # find ATTN_TRACE block
            am = re.search(r"\[ATTN_TRACE\] Layer-0 per-head Q·K raw dots:\n(.*?)(?:\n\n|$)", block, re.S)
            if am:
                head_lines = am.group(1).strip().splitlines()
                heads = {}
                for hl in head_lines:
                    m2 = re.search(r"Head\s*(\d+):\s*raw_dot=([0-9.eE+-]+)(?:\s+head_scalar=([0-9.eE+-]+)\s+scaled=([0-9.eE+-]+))?", hl)
                    if m2:
                        idx = int(m2.group(1))
                        raw = float(m2.group(2))
                        head_scalar = float(m2.group(3)) if m2.group(3) else None
                        scaled = float(m2.group(4)) if m2.group(4) else (raw * head_scalar if head_scalar is not None else None)
                        heads[idx] = {'raw': raw, 'head_scalar': head_scalar, 'scaled': scaled}
                # convert to ordered list
                if heads:
                    max_h = max(heads.keys())
                    vals = [heads.get(i, {'raw': float('nan'), 'head_scalar': None, 'scaled': None}) for i in range(max_h+1)]
                    steps[step] = vals
        return steps

    def parse_sapphire_attn(path):
        with open(path, 'r') as f:
            s = f.read()
        # parse header lines with head_scalar and token-0 raw/scaled
        heads = {}
        for m in re.finditer(r"ATTN_TRACE: L0 H(\d+) head_scalar=([0-9.eE+-]+) raw_dot\[0\]=([0-9.eE+-]+) scaled\[0\]=([0-9.eE+-]+)", s):
            hi = int(m.group(1))
            head_scalar = float(m.group(2))
            raw = float(m.group(3))
            scaled = float(m.group(4))
            heads.setdefault(hi, {})
            heads[hi].update({'raw': raw, 'head_scalar': head_scalar, 'scaled': scaled})
        # parse per-token lines t=0..N (if present)
        for m in re.finditer(r"ATTN_TRACE: L0 H(\d+) t=(\d+) raw=([0-9.eE+-]+).*?scaled=([0-9.eE+-]+)", s):
            hi = int(m.group(1))
            t = int(m.group(2))
            raw = float(m.group(3))
            scaled = float(m.group(4))
            heads.setdefault(hi, {})
            heads[hi].setdefault('tokens', {})
            heads[hi]['tokens'][t] = {'raw': raw, 'scaled': scaled}
        if not heads:
            return {}
        # return as step 0 only, build ordered list
        max_h = max(heads.keys())
        vals = [heads.get(i, {'raw': float('nan'), 'head_scalar': None, 'scaled': None, 'tokens': {}}) for i in range(max_h+1)]
        return {0: vals}

    torch_attn = parse_torch_attn(torch_log)
    sapphire_attn = parse_sapphire_attn(sapp_log)

    if torch_attn or sapphire_attn:
        print('\nLayer-0 per-head Q\u00B7K comparison (raw & scaled, token 0):')
        # choose step 0 where possible
        step = 0
        tvals = torch_attn.get(step, [])
        sval = sapphire_attn.get(0, [])
        maxh = max(len(tvals), len(sval))
        print('Head | Py_raw | Sap_raw | Raw_diff | Py_scaled | Sap_scaled | Scaled_diff')
        print('---- | ------:| -------:| --------:| ---------:| ---------:| ----------:')
        raw_diffs = []
        scaled_diffs = []
        for h in range(maxh):
            tentry = tvals[h] if h < len(tvals) else {'raw': float('nan'), 'scaled': None}
            sentry = sval[h] if h < len(sval) else {'raw': float('nan'), 'scaled': None}
            t_raw = tentry.get('raw') if isinstance(tentry, dict) else tentry
            t_scaled = tentry.get('scaled') if isinstance(tentry, dict) else None
            s_raw = sentry.get('raw') if isinstance(sentry, dict) else sentry
            s_scaled = sentry.get('scaled') if isinstance(sentry, dict) else None
            raw_diff = (t_raw - s_raw) if (t_raw is not None and s_raw is not None) else float('nan')
            if t_scaled is None and tentry.get('head_scalar') is not None and t_raw is not None:
                t_scaled = t_raw * tentry.get('head_scalar')
            if s_scaled is None and sentry.get('head_scalar') is not None and s_raw is not None:
                s_scaled = s_raw * sentry.get('head_scalar')
            scaled_diff = (t_scaled - s_scaled) if (t_scaled is not None and s_scaled is not None) else float('nan')
            raw_diffs.append(raw_diff)
            scaled_diffs.append(scaled_diff)
            def fmt(x):
                return f"{x:.6f}" if (x is not None and not (isinstance(x, float) and (x != x))) else 'N/A'
            print(f"{h:4d} | {fmt(t_raw):>7} | {fmt(s_raw):>7} | {fmt(raw_diff):>8} | {fmt(t_scaled):>9} | {fmt(s_scaled):>9} | {fmt(scaled_diff):>10}")
        # compute simple stats
        import math
        valid_raw = [x for x in raw_diffs if x == x]
        valid_scaled = [x for x in scaled_diffs if x == x]
        if valid_raw:
            mean_abs_raw = sum(abs(x) for x in valid_raw) / len(valid_raw)
            rmse_raw = math.sqrt(sum(x*x for x in valid_raw) / len(valid_raw))
            print(f"\nRaw diffs — mean_abs={mean_abs_raw:.6f} rmse={rmse_raw:.6f}")
        if valid_scaled:
            mean_abs_scaled = sum(abs(x) for x in valid_scaled) / len(valid_scaled)
            rmse_scaled = math.sqrt(sum(x*x for x in valid_scaled) / len(valid_scaled))
            print(f"Scaled diffs — mean_abs={mean_abs_scaled:.6f} rmse={rmse_scaled:.6f}")
    else:
        print('\nNo attention traces found in either log.')

if __name__ == '__main__':
    main()
