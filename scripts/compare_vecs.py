#!/usr/bin/env python3
import re
import sys
import math
from collections import defaultdict

def parse_vecs(path):
    data = defaultdict(lambda: {'Q': {}, 'K': {}})
    pattern = re.compile(r'ATTN_VEC:\s+L(\d+)\s+H(\d+)\s+.*?(Q_vec|K_vec):\s*(.*)')
    with open(path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            L = int(m.group(1))
            H = int(m.group(2))
            typ = m.group(3)
            vals = [float(x) for x in m.group(4).strip().split() if x]
            data[L][typ.replace('_vec','')][H] = vals
    return data

def compare(a, b):
    layers = sorted(set(list(a.keys()) + list(b.keys())))
    for L in layers:
        heads = sorted(set(list(a[L]['Q'].keys()) + list(b[L]['Q'].keys())))
        print(f"\nLayer {L} Q-vector comparisons:")
        for h in heads:
            qa = a[L]['Q'].get(h)
            qb = b[L]['Q'].get(h)
            if qa is None or qb is None:
                print(f"  Head {h}: missing Q in one of the logs")
                continue
            n = min(len(qa), len(qb))
            diffs = [(qa[i]-qb[i]) for i in range(n)]
            mean_abs = sum(abs(x) for x in diffs)/n
            rmse = math.sqrt(sum(x*x for x in diffs)/n)
            print(f"  Head {h}: len={n} mean_abs={mean_abs:.6f} rmse={rmse:.6f}")

        print(f"\nLayer {L} K-vector comparisons:")
        heads = sorted(set(list(a[L]['K'].keys()) + list(b[L]['K'].keys())))
        for h in heads:
            ka = a[L]['K'].get(h)
            kb = b[L]['K'].get(h)
            if ka is None or kb is None:
                print(f"  Head {h}: missing K in one of the logs")
                continue
            n = min(len(ka), len(kb))
            diffs = [(ka[i]-kb[i]) for i in range(n)]
            mean_abs = sum(abs(x) for x in diffs)/n
            rmse = math.sqrt(sum(x*x for x in diffs)/n)
            print(f"  Head {h}: len={n} mean_abs={mean_abs:.6f} rmse={rmse:.6f}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: compare_vecs.py sapphire_log torch_log')
        sys.exit(2)
    sap = parse_vecs(sys.argv[1])
    tor = parse_vecs(sys.argv[2])
    compare(tor, sap)
