# -*- coding: utf-8 -*-
"""e7 评估：未见问法采样，看是真接话还是背原句。

`--memory` 时先检索外部库，把 top1 答案拼成前一轮再生成，并分两组报告：
库内问题（验证"检索→落地"机制）与库外问题（看是否诚实失败）。
"""
from __future__ import annotations

import os
import random

import torch

from ..vocab import A, BOS, U
from .sample import (build_retrieval, ckpt_path, load_memory_pairs, load_model, reply,
                     retrieval_prefix)

PROMPTS = [
    "你是谁",
    "今天心情不好",
    "讲个笑话",
    "明天会下雨吗",
    "帮我写一首诗",
    "人生的意义是什么",
    "张伟住在哪",
    "谢谢你",
]


def run(max_new: int = 60, temp: float = 0.8, ckpt: str | None = None,
        memory: bool = False, n_mem: int = 6) -> None:
    model, vocab = load_model(ckpt)
    path = ckpt_path(ckpt)
    step = torch.load(path, map_location="cpu", weights_only=False).get("step")
    print(f"[eval] ckpt step={step} mode={'memory' if memory else 'plain'}")

    if memory:
        pairs = load_memory_pairs()
        mem = build_retrieval(pairs)
        lib = random.Random(0).sample(pairs, min(n_mem, len(pairs)))
        print("== 库内问题（验证外部事实记忆→落地） ==")
        exact = 0
        for q, gold in lib:
            prefix, ret = retrieval_prefix(vocab, mem, q)
            new, conf = reply(model, vocab,
                              [BOS, *prefix, U, *vocab.encode(q), A],
                              max_new, temp)
            out = vocab.decode(new).strip()
            gold = gold.strip()
            exact += int(out == gold)
            print(f"问: {q}")
            print(f"记: {ret}")
            print(f"它: {out}")
            print(f"答: {gold}")
            print(f"    (置信={conf:.2f}, 逐字={out == gold})")
        print(f"[eval] 库内逐字命中 {exact}/{len(lib)}")
        print("== 库外问题（诚实失败） ==")
        for q in PROMPTS:
            prefix, ret = retrieval_prefix(vocab, mem, q)
            new, conf = reply(model, vocab,
                              [BOS, *prefix, U, *vocab.encode(q), A],
                              max_new, temp)
            print(f"你: {q}")
            print(f"记: {ret if ret else '无命中'}")
            print(f"它: {vocab.decode(new)}")
            print(f"    (置信={conf:.2f})")
        return

    for q in PROMPTS:
        new, conf = reply(model, vocab, [BOS, U, *vocab.encode(q), A], max_new, temp)
        print(f"你: {q}")
        print(f"它: {vocab.decode(new)}")
        print(f"    (置信={conf:.2f})")
