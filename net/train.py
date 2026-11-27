# -*- coding: utf-8 -*-
"""e7 两段训练：语料预热 → 对话联合。"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch

from .. import corpus as C
from ..vocab import Vocab
from .brain import (BrainNet, Config, conf_loss, intent_loss, lm_loss,
                    load_state_compat, role_loss)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.environ.get("E7_ARTIFACTS", os.path.join(HERE, "artifacts"))
SMALL = {"d": 96, "n_layer": 2, "n_head": 2, "ctx": 64}
BIG = {"d": 384, "n_layer": 6, "n_head": 6, "ctx": 192}


def _paths() -> dict:
    return {
        "vocab": os.path.join(ART, "vocab.json"),
        "corpus": os.path.join(ART, "corpus.npy"),
        "dialogues": os.path.join(ART, "dialogues.pt"),
        "dialogues_mem": os.path.join(ART, "dialogues_mem.pt"),
        "dialogues_rec": os.path.join(ART, "dialogues_rec.pt"),
        "ckpt": os.path.join(ART, "ckpt.pt"),
        "metrics": os.path.join(ART, "metrics.jsonl"),
    }


def prep(max_chars: int = 2_000_000, min_freq: int = 2,
         dialogues_only: bool = False, corpus_only: bool = False,
         memory: bool = False, reconstruct: bool = False,
         bpe: int = 0) -> None:
    os.makedirs(ART, exist_ok=True)
    p = _paths()
    if dialogues_only:
        vocab = Vocab.load(p["vocab"])
        if reconstruct:
            samples = C.build_reconstruct_dialogues(vocab, ctx=192)
            torch.save(samples, p["dialogues_rec"])
            n_span = sum(int((s["gt_gate"] > 0).sum()) for s in samples)
            n_lm = sum(int(s["lm"].sum()) for s in samples)
            print(f"[prep] reconstruct dialogues={len(samples)} "
                  f"span_ratio={n_span / max(n_lm, 1):.3f} "
                  f"(vocab kept {len(vocab)})")
            return
        if memory:
            samples = C.build_memory_dialogues(vocab, ctx=192)
            torch.save(samples, p["dialogues_mem"])
            print(f"[prep] memory dialogues={len(samples)} "
                  f"(vocab kept {len(vocab)})")
            return
        samples = C.build_dialogues(vocab, ctx=192)
        torch.save(samples, p["dialogues"])
        print(f"[prep] dialogues={len(samples)} (vocab kept {len(vocab)})")
        return
    if corpus_only:
        if bpe:
            print(f"[prep] 训练 BPE 词表 vocab={bpe}", flush=True)
            vocab = C.train_bpe(C.CORPUS_SRC, vocab_size=bpe, out=ART,
                                max_chars=max_chars)
            print(f"[prep] BPE vocab={len(vocab)}")
            return
        vocab = Vocab.load(p["vocab"])
        arr = C.build_corpus(vocab, max_chars=max_chars)
        np.save(p["corpus"], arr)
        print(f"[prep] corpus={len(arr)} tokens (vocab kept {len(vocab)})")
        return
    if bpe:
        print(f"[prep] 训练 BPE 词表 vocab={bpe}", flush=True)
        vocab = C.train_bpe(C.CORPUS_SRC, vocab_size=bpe, out=ART,
                            max_chars=max_chars)
        print(f"[prep] BPE vocab={len(vocab)}", flush=True)
    else:
        texts: list[str] = []
        seen = 0
        mark = 50_000_000
        print(f"[prep] 读语料 {C.CORPUS_SRC}", flush=True)
        for t in C.iter_texts(C.CORPUS_SRC, max_chars):
            texts.append(t)
            seen += len(t)
            if seen >= mark:
                print(f"[prep] 已读 {seen / 1e6:.0f}M chars", flush=True)
                mark += 50_000_000
            if seen >= max_chars:
                break
        print(f"[prep] 读对话 {C.MYQA_SRC}", flush=True)
        convs = C.load_conversations()
        texts.extend(content for turns in convs for _, content in turns)
        print(f"[prep] 建词表（{len(texts)} 段，{seen} chars）", flush=True)
        vocab = Vocab.build(texts, min_freq=min_freq)
        vocab.save(p["vocab"])
        del texts
    print(f"[prep] 编码语料 -> corpus.npy", flush=True)

    arr = C.build_corpus(vocab, max_chars=max_chars)
    np.save(p["corpus"], arr)
    samples = C.build_dialogues(vocab, ctx=192)
    torch.save(samples, p["dialogues"])
    print(f"[prep] vocab={len(vocab)} corpus={len(arr)} chars "
          f"dialogues={len(samples)} ctx<=192")


def _cfg(vocab: Vocab, small: bool, n_intent: int, big: bool = False,
         shape: dict | None = None) -> Config:
    cfg = Config(vocab_size=len(vocab), n_intent=max(6, n_intent))
    preset = SMALL if small else (BIG if big else None)
    if preset:
        for k, v in preset.items():
            setattr(cfg, k, v)
    for k, v in (shape or {}).items():
        setattr(cfg, k, v)
    if cfg.d % cfg.n_head:
        raise ValueError(f"d={cfg.d} 不能被 n_head={cfg.n_head} 整除")
    return cfg


def _corpus_batch(arr: np.ndarray, ctx: int, bs: int):
    hi = len(arr) - ctx - 1
    starts = np.random.randint(0, hi, size=bs)
    x = np.stack([arr[s:s + ctx] for s in starts]).astype(np.int64)
    y = np.stack([arr[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
    return (torch.from_numpy(x), torch.from_numpy(y),
            torch.ones_like(torch.from_numpy(y), dtype=torch.float32),
            torch.full_like(torch.from_numpy(y), -100))


def _dialogue_batch(samples: list[dict], ctx: int, bs: int,
                    align: bool = False):
    pick = random.choices(samples, k=bs)
    T = min(ctx, max(len(s["x"]) for s in pick))
    x = np.zeros((bs, T), dtype=np.int64)
    y = np.zeros((bs, T), dtype=np.int64)
    m = np.zeros((bs, T), dtype=np.float32)
    r = np.full((bs, T), -100, dtype=np.int64)
    al = np.full((bs, T), -100, dtype=np.int64)
    gg = np.full((bs, T), -100.0, dtype=np.float32)
    for i, s in enumerate(pick):
        k = min(T, len(s["x"]))
        x[i, :k] = s["x"][:k]
        y[i, :k] = s["y"][:k]
        m[i, :k] = s["lm"][:k]
        r[i, :k] = s["role"][:k]
        if align:
            al[i, :k] = s["align"][:k]
            gg[i, :k] = s["gt_gate"][:k]
    out = (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m),
           torch.from_numpy(r))
    if align:
        return (*out, torch.from_numpy(al), torch.from_numpy(gg))
    return out


def _intent_batch(rows: list[dict], ctx: int, bs: int):
    pick = random.choices(rows, k=bs)
    T = min(ctx, max(len(r["ids"]) for r in pick))
    x = np.zeros((len(pick), T), dtype=np.int64)
    for i, r in enumerate(pick):
        ids = r["ids"][:T]
        x[i, :len(ids)] = ids
    y = torch.tensor([r["label"] for r in pick], dtype=torch.long)
    return torch.from_numpy(x), y


def _step_loss(model: BrainNet, x, y, lm, role,
               intents=None, w_role=0.3, w_conf=0.1, w_intent=0.3,
               w_mem=0.1, align=None, gt_gate=None) -> tuple:
    dev = next(model.parameters()).device
    x, y, lm, role = x.to(dev), y.to(dev), lm.to(dev), role.to(dev)
    if align is not None:
        align = align.to(dev)
    if gt_gate is not None:
        gt_gate = gt_gate.to(dev)
    out = model(x)
    l_lm, _ = lm_loss(out, y, lm, x, align=align, gt_gate=gt_gate)
    l_role, role_stats = role_loss(out, role)
    l_conf, conf_stats = conf_loss(out, y, lm)
    # 记忆辅助目标：让"记忆读出"本身可解码为下一个字，逼语义记忆真正承载信息
    # （外部事实由 retrieval 提供，这里训的是学习型语义记忆）
    l_mem, _ = lm_loss({"logits": model.lm(out["mem"])}, y, lm)
    total = l_lm + w_role * l_role + w_conf * l_conf + w_mem * l_mem
    stats = {"lm": float(l_lm), "conf": float(l_conf), "mem": float(l_mem),
             "gate": float(out["gate"].mean()),
             **conf_stats, **role_stats}
    if gt_gate is not None:
        gt = gt_gate.to(dev)
        sp, fr = gt > 0.5, gt == 0
        if sp.any():
            stats["gate_span"] = float(out["gate"][sp].mean())
        if fr.any():
            stats["gate_free"] = float(out["gate"][fr].mean())
    if intents is not None and len(intents[0]) > 0:
        xi, yi = intents
        oi = model(xi.to(dev))
        l_i, acc_i = intent_loss(oi, yi.to(dev))
        total = total + w_intent * l_i
        stats.update({"intent": float(l_i), "intent_acc": acc_i})
    return total, stats


def _save(model: BrainNet, opt, cfg: Config, step: int, path: str,
          quiet: bool = False) -> None:
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"cfg": cfg.to_dict(), "state": state,
                "opt": opt.state_dict(), "step": step}, path)
    if not quiet:
        print(f"[save] {path} step={step}")


def train(stage: str = "lm", steps: int = 300, bs: int = 16, lr: float = 3e-4,
          small: bool = False, resume: bool = False, seed: int = 0,
          save_every: int = 200, big: bool = False,
          memory: bool = False, reconstruct: bool = False,
          shape: dict | None = None) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    p = _paths()
    vocab = Vocab.load(p["vocab"])
    arr = np.load(p["corpus"])
    key = "dialogues_rec" if reconstruct else \
        ("dialogues_mem" if memory else "dialogues")
    samples = torch.load(p[key], weights_only=False)
    # 重构训练时混入普通对话：否则模型只见过「检索轮 + 问答轮」，
    # 不带检索的 chat 变成分布外，会胡言乱语。
    plain = torch.load(p["dialogues"], weights_only=False) if reconstruct else []
    intents, labels = C.load_intents(vocab)

    cfg = _cfg(vocab, small, len(labels), big, shape)
    if memory and cfg.ctx < 192:
        raise ValueError("检索记忆样本按 ctx=192 构建，--memory 不能配 --small")
    if reconstruct and cfg.ctx < 192:
        raise ValueError("重构样本按 ctx=192 构建，--reconstruct 不能配 --small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainNet(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    start = 0
    if resume and os.path.exists(p["ckpt"]):
        ck = torch.load(p["ckpt"], map_location="cpu", weights_only=False)
        missing = load_state_compat(model, ck["state"])
        if missing:
            print(f"[resume] 新参数按初始化值加载: {missing}")
        if "opt" in ck:
            try:
                opt.load_state_dict(ck["opt"])
                for st in opt.state.values():
                    for k2, v2 in st.items():
                        if torch.is_tensor(v2):
                            st[k2] = v2.to(device)
            except ValueError as e:
                # 新增参数（如 copy_head）后参数组对不上：只能重新累积动量
                print(f"[resume] 优化器状态不兼容，重新开始累积: {e}")
        # ckpt 里存着旧的 lr，续训时以命令行为准（否则 --lr 形同虚设）
        for g in opt.param_groups:
            g["lr"] = lr
        start = ck.get("step", 0)
        print(f"[resume] step={start}")

    n_params = sum(q.numel() for q in model.parameters())
    print(f"[train] stage={stage} steps={steps} bs={bs} lr={lr} "
          f"d={cfg.d} L={cfg.n_layer} ctx={cfg.ctx} params={n_params/1e6:.2f}M "
          f"device={device.type}")

    model.train()
    t0 = time.time()
    running = None
    for i in range(1, steps + 1):
        if stage == "lm":
            x, y, m, r = _corpus_batch(arr, cfg.ctx, bs)
            total, stats = _step_loss(model, x, y, m, r)
            stats["bpc"] = stats["lm"] / 0.6931
        else:
            half = max(1, bs // 2)
            if reconstruct:
                n_plain = max(1, bs // 4)
                xd, yd, md, rd, ad, gd = _dialogue_batch(
                    samples, cfg.ctx, max(1, half - n_plain), align=True)
                td, sd = _step_loss(model, xd, yd, md, rd,
                                    align=ad, gt_gate=gd)
                xp, yp, mp, rp = _dialogue_batch(plain, cfg.ctx, n_plain)
                tp, sp = _step_loss(model, xp, yp, mp, rp)
                td = td + tp
                sd["lm_plain"] = sp["lm"]
            else:
                xd, yd, md, rd = _dialogue_batch(samples, cfg.ctx, half)
                td, sd = _step_loss(model, xd, yd, md, rd)
            xc, yc, mc, rc = _corpus_batch(arr, cfg.ctx, bs - half)
            tc, sc = _step_loss(model, xc, yc, mc, rc)
            total = td + tc
            # 角色/置信只取对话批：语料批没有角色标签、置信口径也不同，
            # 之前把两批平均会把 role_acc 稀释一半。
            stats = dict(sd)
            lm_parts = [sd["lm"], sc["lm"]]
            if reconstruct:
                lm_parts.append(sd["lm_plain"])
            stats["lm"] = sum(lm_parts) / len(lm_parts)
            stats["lm_dlg"] = sd["lm"]
            stats["lm_corpus"] = sc["lm"]
            stats["bpc"] = stats["lm"] / 0.6931
            if intents:
                xi, yi = _intent_batch(intents, cfg.ctx, max(4, bs // 4))
                oi = model(xi.to(device))
                li, ai = intent_loss(oi, yi.to(device))
                total = total + 0.3 * li
                stats.update({"intent": float(li), "intent_acc": ai})

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running = stats if running is None else \
            {k: running[k] * 0.9 + stats.get(k, 0.0) * 0.1 for k in running}
        if i % 10 == 0 or i == steps:
            msg = " ".join(f"{k}={v:.3f}" for k, v in running.items())
            print(f"[{i:04d}/{steps}] {msg} ({time.time() - t0:.0f}s)")
            with open(p["metrics"], "a", encoding="utf-8") as f:
                f.write(json.dumps({"stage": stage, "step": start + i,
                                    "d": cfg.d, "L": cfg.n_layer,
                                    **{k: round(float(v), 4) for k, v in running.items()}},
                                   ensure_ascii=False) + "\n")
        if i % save_every == 0 or i == steps:
            _save(model, opt, cfg, start + i, p["ckpt"], quiet=(i != steps))
