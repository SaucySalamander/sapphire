import math
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Key:
    layer: int
    head: int
    token: int
    kind: str  # 'Q' or 'K'


ATTN_RE = re.compile(
    r"^ATTN_VEC: L(?P<layer>\d+) H(?P<head>\d+) token=(?P<token>\d+) (?P<kind>[QK])_vec:\s*(?P<vals>.*)$"
)


def parse_attn_vecs(path: str) -> Dict[Key, List[float]]:
    out: Dict[Key, List[float]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = ATTN_RE.match(line)
            if not m:
                continue
            layer = int(m.group("layer"))
            head = int(m.group("head"))
            token = int(m.group("token"))
            kind = m.group("kind")
            vals_s = m.group("vals").strip()
            if not vals_s:
                vals = []
            else:
                vals = [float(x) for x in vals_s.split()]
            out[Key(layer=layer, head=head, token=token, kind=kind)] = vals
    return out


def rmse(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    s = 0.0
    for i in range(n):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s / n)


def max_abs(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    m = 0.0
    for i in range(n):
        m = max(m, abs(a[i] - b[i]))
    return m


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print("usage: python -u models/compare_attn_vecs.py <torch_log> <sapphire_log>")
        return 2

    torch_path, sapphire_path = argv[1], argv[2]
    t = parse_attn_vecs(torch_path)
    s = parse_attn_vecs(sapphire_path)

    keys = sorted(set(t.keys()) & set(s.keys()), key=lambda k: (k.layer, k.head, k.token, k.kind))
    if not keys:
        print("No overlapping ATTN_VEC entries found.")
        return 1

    print("layer head token kind n rmse max_abs")
    worst: Tuple[float, Key] = (-1.0, keys[0])

    for k in keys:
        tv = t[k]
        sv = s[k]
        n = min(len(tv), len(sv))
        r = rmse(tv, sv)
        m = max_abs(tv, sv)
        print(f"{k.layer:5d} {k.head:4d} {k.token:5d} {k.kind:4s} {n:3d} {r:12.6g} {m:12.6g}")
        if math.isfinite(r) and r > worst[0]:
            worst = (r, k)

    wr, wk = worst
    print(f"\nWORST: L{wk.layer} H{wk.head} token={wk.token} {wk.kind}_vec rmse={wr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
