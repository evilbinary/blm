# -*- coding: utf-8 -*-
"""e7 CLI: prep / train / chat / info。"""
from __future__ import annotations

import argparse
import os

from .net import eval as evaluator
from .net import sample, train as trainer


def main() -> None:
    p = argparse.ArgumentParser(prog="e7", description="e7 可训练功能链")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prep", help="构建词表与数据缓存")
    pp.add_argument("--max-chars", type=int, default=2_000_000)
    pp.add_argument("--min-freq", type=int, default=2)
    pp.add_argument("--dialogues-only", action="store_true",
                    help="只重建对话缓存，保持现有词表")
    pp.add_argument("--corpus-only", action="store_true",
                    help="只重建语料缓存，保持现有词表")
    pp.add_argument("--memory", action="store_true",
                    help="构建检索式记忆样本 dialogues_mem.pt（配合 --dialogues-only）")
    pp.add_argument("--reconstruct", action="store_true",
                    help="构建重构样本 dialogues_rec.pt（近似问法 + span 对齐）")

    pt = sub.add_parser("train", help="训练")
    pt.add_argument("--stage", choices=["lm", "joint"], default="lm")
    pt.add_argument("--steps", type=int, default=300)
    pt.add_argument("--bs", type=int, default=16)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--small", action="store_true")
    pt.add_argument("--big", action="store_true", help="d=384 / 6 层")
    pt.add_argument("--resume", action="store_true")
    pt.add_argument("--save-every", type=int, default=200)
    pt.add_argument("--memory", action="store_true",
                    help="用检索上下文样本（dialogues_mem.pt）训练")
    pt.add_argument("--reconstruct", action="store_true",
                    help="用重构样本（dialogues_rec.pt）训练，带 span 对齐监督")

    pc = sub.add_parser("chat", help="对话")
    pc.add_argument("--temp", type=float, default=0.8)
    pc.add_argument("--max-new", type=int, default=80)
    pc.add_argument("--top-k", type=int, default=40)
    pc.add_argument("--top-p", type=float, default=0.9)
    pc.add_argument("--rep-penalty", type=float, default=1.3)
    pc.add_argument("--no-repeat", type=int, default=3)
    pc.add_argument("--ckpt", default=None)
    pc.add_argument("--memory", action="store_true",
                    help="推理前先检索外部库")

    sub.add_parser("info", help="查看产物")
    pe = sub.add_parser("eval", help="未见问法采样")
    pe.add_argument("--max-new", type=int, default=60)
    pe.add_argument("--temp", type=float, default=0.8)
    pe.add_argument("--ckpt", default=None)
    pe.add_argument("--memory", action="store_true",
                    help="检索式评估：库内问题 + 库外问题")
    pe.add_argument("--n-mem", type=int, default=6,
                    help="库内抽检条数")

    a = p.parse_args()
    if a.cmd == "prep":
        trainer.prep(a.max_chars, a.min_freq, a.dialogues_only, a.corpus_only,
                     a.memory, a.reconstruct)
    elif a.cmd == "train":
        trainer.train(a.stage, a.steps, a.bs, a.lr, a.small, a.resume,
                      save_every=a.save_every, big=a.big, memory=a.memory,
                      reconstruct=a.reconstruct)
    elif a.cmd == "chat":
        sample.chat(a.temp, a.max_new, a.ckpt, a.top_k, a.top_p,
                    a.rep_penalty, a.no_repeat, a.memory)
    elif a.cmd == "eval":
        evaluator.run(a.max_new, a.temp, a.ckpt, a.memory, a.n_mem)
    elif a.cmd == "info":
        for name in ("vocab.json", "corpus.npy", "dialogues.pt",
                     "dialogues_mem.pt", "dialogues_rec.pt", "ckpt.pt"):
            path = os.path.join(trainer.ART, name)
            if os.path.exists(path):
                print(f"{name:<16} {os.path.getsize(path)/1e6:.2f} MB")
            else:
                print(f"{name:<16} -")


if __name__ == "__main__":
    main()
