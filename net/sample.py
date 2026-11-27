# -*- coding: utf-8 -*-
"""e7 采样与对话：加载训练产物，按 <u>...<a> 格式续写。"""
from __future__ import annotations

import os
import time

import torch

from .. import corpus as C
from ..vocab import A, BOS, EOS, U, Vocab
from .brain import BrainNet, Config, load_state_compat
from .retrieval import RetrievalMemory

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts")


def load_memory_pairs() -> list[tuple[str, str]]:
    """外部记忆库的 (question, answer)：与训练数据同源（my.jsonl 采样结果）。"""
    return C.load_qa_pairs()


def build_retrieval(pairs: list[tuple[str, str]] | None = None,
                    max_answer: int = 60) -> RetrievalMemory:
    return RetrievalMemory(pairs if pairs is not None else load_memory_pairs(),
                           max_answer=max_answer)


def retrieval_prefix(vocab: Vocab, mem: RetrievalMemory, query: str,
                     k: int = 1) -> tuple[list[int], str | None]:
    """推理时的记忆轮：`<u>问<a>检索答案<eos>`，拼在真正问答之前。"""
    hits = mem.search(query, k=k)
    if not hits:
        return [], None
    ret = hits[0][1]
    return [U, *vocab.encode(query), A, *vocab.encode(ret), EOS], ret


def load_model(path: str | None = None) -> tuple[BrainNet, Vocab]:
    path = path or os.path.join(ART, "ckpt.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"没有训练产物: {path}，先跑 train")
    vocab = Vocab.load(os.path.join(ART, "vocab.json"))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainNet(Config(**ck["cfg"])).to(device)
    load_state_compat(model, ck["state"])
    model.eval()
    return model, vocab


def stream_reply(model: BrainNet, vocab: Vocab, history: list[int],
                 max_new: int = 80, temp: float = 0.8, top_k: int = 40,
                 top_p: float = 0.9, rep_penalty: float = 1.3,
                 no_repeat: int = 3):
    """逐 token 生成：yield 新 token，结束时 return 最后置信。

    与 reply 共用同一套采样逻辑；chat 直接消费它做流式打印。
    含重复惩罚 + no-repeat n-gram + nucleus，抑制"嘻嘻嘻""雪雪雪"式复读。
    """
    with torch.no_grad():
        ids = list(history)[-model.cfg.ctx:]
        dev = next(model.parameters()).device
        conf = 0.0
        gate = 0.0
        for _ in range(max_new):
            x = torch.tensor([ids[-model.cfg.ctx:]], dtype=torch.long, device=dev)
            res = model(x)
            conf = float(res["conf"][0, -1])
            gate = float(res["gate"][0, -1])
            logits = res["logits"][0, -1].clone()
            if rep_penalty and rep_penalty != 1.0:
                for t in set(ids[-model.cfg.ctx:]):
                    logits[t] = logits[t] / rep_penalty if logits[t] > 0 \
                        else logits[t] * rep_penalty
            if no_repeat and len(ids) >= no_repeat:
                prefix = tuple(ids[-(no_repeat - 1):])
                for i in range(len(ids) - no_repeat + 1):
                    if tuple(ids[i:i + no_repeat - 1]) == prefix:
                        logits[ids[i + no_repeat - 1]] = float("-inf")
            logits = logits / max(temp, 1e-3)
            probs = torch.softmax(logits, -1)
            if gate > 1e-4:               # 复制门控：混合指针分布
                pcopy = torch.zeros_like(probs)
                pcopy.scatter_add_(0, x[0], res["copy_attn"][0, -1])
                probs = gate * pcopy + (1.0 - gate) * probs
            if top_k:
                v, idx = torch.topk(probs, min(top_k, probs.numel()))
                keep = torch.zeros_like(probs)
                keep[idx] = 1.0
                probs = probs * keep
            if top_p and 0 < top_p < 1:
                sp, si = torch.sort(probs, descending=True)
                sp[torch.cumsum(sp, 0) - sp > top_p] = 0.0
                probs = torch.zeros_like(probs)
                probs[si] = sp
            probs = probs / probs.sum().clamp_min(1e-9)
            tok = int(torch.multinomial(probs, 1))
            if tok == EOS:
                return conf
            ids.append(tok)
            yield tok
        return conf


@torch.no_grad()
def reply(model: BrainNet, vocab: Vocab, history: list[int],
          max_new: int = 80, temp: float = 0.8, top_k: int = 40,
          top_p: float = 0.9, rep_penalty: float = 1.3,
          no_repeat: int = 3) -> tuple[list[int], float]:
    """history 已包含 <u>问题<a>；返回新 token 与最后位置置信。"""
    out: list[int] = []
    conf = 0.0
    gen = stream_reply(model, vocab, history, max_new, temp, top_k,
                       top_p, rep_penalty, no_repeat)
    while True:
        try:
            out.append(next(gen))
        except StopIteration as e:
            conf = float(e.value or 0.0)
            break
    return out, conf


def chat(temp: float = 0.8, max_new: int = 80, ckpt: str | None = None,
         top_k: int = 40, top_p: float = 0.9, rep_penalty: float = 1.3,
         no_repeat: int = 3, memory: bool = False) -> None:
    model, vocab = load_model(ckpt)
    mem = build_retrieval() if memory else None
    print(f"[e7] 功能链对话 | d={model.cfg.d} L={model.cfg.n_layer} "
          f"ctx={model.cfg.ctx} | 记忆={'检索' if memory else '槽位'} | q 退出")
    history: list[int] = [BOS]
    while True:
        try:
            line = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("q", "quit", "exit"):
            break
        if not line:
            continue
        if mem is not None:
            prefix, ret = retrieval_prefix(vocab, mem, line)
            history = [BOS, *prefix]          # 检索模式下每轮重置，对齐训练分布
            print(f"    (检索: {ret if ret else '无命中'})")
        history += [U, *vocab.encode(line), A]
        new: list[int] = []
        conf = 0.0
        t0 = time.perf_counter()
        t_first = 0.0
        print("它: ", end="", flush=True)
        gen = stream_reply(model, vocab, history, max_new, temp, top_k,
                           top_p, rep_penalty, no_repeat)
        while True:
            try:
                tok = next(gen)
            except StopIteration as e:
                conf = float(e.value or 0.0)
                break
            if not new:
                t_first = time.perf_counter() - t0
            new.append(tok)
            s = vocab.decode([tok])
            if s:
                print(s, end="", flush=True)
        dt = time.perf_counter() - t0
        print()
        history += new + [EOS]
        history = history[-model.cfg.ctx:]
        n = len(new)
        speed = n / max(dt, 1e-9)
        ms = dt / max(n, 1) * 1000
        print(f"    (置信={conf:.2f}, tokens={n}, {dt:.2f}s, "
              f"{speed:.1f} tok/s, {ms:.0f} ms/token, 首字={t_first:.2f}s)")
