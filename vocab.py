# -*- coding: utf-8 -*-
"""e7 字符词表。"""
from __future__ import annotations

import json
import os
from collections import Counter

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<u>", "<a>"]
PAD, UNK, BOS, EOS, U, A = range(len(SPECIALS))


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {c: i for i, c in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts, min_freq: int = 2) -> "Vocab":
        cnt: Counter[str] = Counter()
        for t in texts:
            cnt.update(t)
        chars = [c for c, n in cnt.most_common() if n >= min_freq]
        return cls(SPECIALS + chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, UNK) for c in text]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i < len(SPECIALS):
                continue
            out.append(self.itos[i])
        return "".join(out)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["itos"])
