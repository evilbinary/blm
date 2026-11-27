# -*- coding: utf-8 -*-
"""e7 词表：字符级（默认）与 BPE 两套实现，接口一致。"""
from __future__ import annotations

import json
import os
from collections import Counter

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<u>", "<a>"]
PAD, UNK, BOS, EOS, U, A = range(len(SPECIALS))


class Vocab:
    """字符级词表：1 个 token = 1 个字符。"""

    kind = "char"

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

    def encode_offsets(self, text: str):
        """-> (ids, [(char_start, char_end), ...])，与 ids 等长。"""
        ids = self.encode(text)
        return ids, [(i, i + 1) for i in range(len(ids))]

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
            json.dump({"kind": "char", "itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        return load_vocab(path)


class BPEVocab:
    """BPE 词表（HF tokenizers），special 固定占 0–5，其余 token 从 6 开始。"""

    kind = "bpe"

    def __init__(self, tok):
        self.tok = tok

    def __len__(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_offsets(self, text: str):
        enc = self.tok.encode(text, add_special_tokens=False)
        return list(enc.ids), [tuple(o) for o in enc.offsets]

    def decode(self, ids) -> str:
        keep = [int(i) for i in ids if int(i) >= len(SPECIALS)]
        return self.tok.decode(keep, skip_special_tokens=True)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tok_path = os.path.join(os.path.dirname(path), "tokenizer.json")
        self.tok.save(tok_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "bpe", "tokenizer": "tokenizer.json",
                       "vocab_size": len(self)}, f, ensure_ascii=False)


def load_vocab(path: str):
    """按 vocab.json 里的 `kind` 自动选择字符级或 BPE。"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("kind") == "bpe":
        from tokenizers import Tokenizer
        p = d.get("tokenizer") or "tokenizer.json"
        if not os.path.isabs(p):
            p = os.path.join(os.path.dirname(path), p)
        return BPEVocab(Tokenizer.from_file(p))
    return Vocab(d["itos"])
