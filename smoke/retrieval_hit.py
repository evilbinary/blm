# -*- coding: utf-8 -*-
"""外部事实记忆冒烟：索引规模、自检索 top1 命中率、几个样例检索。

python -m e7.smoke.retrieval_hit
"""
from __future__ import annotations

import argparse
import random

from .. import corpus as C
from ..net.retrieval import RetrievalMemory


def main() -> None:
    ap = argparse.ArgumentParser(description="e7 检索冒烟")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--max-answer", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pairs = [(t[0][1], t[1][1]) for t in C.load_conversations()
             if len(t) >= 2 and t[0][0] == "user" and t[1][0] == "assistant"]
    mem = RetrievalMemory(pairs, max_answer=a.max_answer)
    idx = random.Random(a.seed).sample(range(len(pairs)),
                                       min(a.samples, len(pairs)))
    hit = 0
    for i in idx:
        q, ans = pairs[i]
        h = mem.search(q, 1)
        hit += bool(h) and h[0][1] == ans[:a.max_answer]
    print(f"[retrieval] size={len(mem)} samples={len(idx)} "
          f"max_ans={a.max_answer} self_top1={hit}/{len(idx)}")
    for q in ("你是谁", "赵丽颖因哪部剧成名？", "光合作用是什么"):
        print(f"[query] {q}")
        for rank, (score, ans) in enumerate(mem.search(q, 2), 1):
            text = ans if len(ans) <= 40 else ans[:40] + "…"
            print(f"  {rank}. {score:.3f}  {text}")


if __name__ == "__main__":
    main()
