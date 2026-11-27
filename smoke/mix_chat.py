# -*- coding: utf-8 -*-
"""混训验证（本地 1.46M）：重构 + 普通对话 + 语料，对比只训重构。

目的：确认"混入普通对话"能把普通模式（无检索）的分布外问题修掉，
同时不破坏重构链路。指标：普通对话留出 CE、重构 span/free 逐字、
语料 bpc、普通模式贪心首字是否又是 `<a>` 自环。

python -m e7.smoke.mix_chat --mix 0 --steps 1200   # 只训 重构+语料（复现现状）
python -m e7.smoke.mix_chat --mix 1 --steps 1200   # 重构+普通+语料（方案）
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..net.brain import BrainNet, Config, lm_loss
from ..net.train import _corpus_batch, _dialogue_batch
from ..vocab import A, BOS, U, Vocab
from .reconstruct import greedy, lcs_len, metrics, mix_probs

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts")
PROMPTS = ["你好，你是谁？", "明天北京天气怎么样？", "给我讲个笑话", "我今天有点累"]


def plain_ce(model: BrainNet, batch) -> float:
    x, y, m, _ = (t for t in batch)
    with torch.no_grad():
        out = model(x)
        ce = F.cross_entropy(out["logits"].reshape(-1, out["logits"].size(-1)),
                             y.reshape(-1), reduction="none").view_as(y)
        return float((ce * m).sum() / m.sum().clamp(min=1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="e7 混训冒烟")
    ap.add_argument("--mix", type=int, choices=[0, 1], default=1)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=192)
    ap.add_argument("--eval-bs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    vocab = Vocab.load(os.path.join(ART, "vocab.json"))
    rec = torch.load(os.path.join(ART, "dialogues_rec.pt"), weights_only=False)
    plain = torch.load(os.path.join(ART, "dialogues.pt"), weights_only=False)
    arr = np.load(os.path.join(ART, "corpus.npy"))
    random.shuffle(rec)
    random.shuffle(plain)
    rec_tr, rec_te = rec[:4000], rec[4000:4400]
    pl_tr, pl_te = plain[:6000], plain[6000:6600]

    cfg = Config(vocab_size=len(vocab), d=a.d, n_layer=a.n_layer,
                 n_head=a.n_head, ctx=a.ctx, dropout=0.0)
    model = BrainNet(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    print(f"[mix] mix={a.mix} steps={a.steps} bs={a.bs} d={a.d} L={a.n_layer} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"rec={len(rec_tr)} plain={len(pl_tr)}")

    rec_b = _dialogue_batch(rec_te, a.ctx, a.eval_bs, align=True)
    pl_b = _dialogue_batch(pl_te, a.ctx, a.eval_bs)
    co_b = _corpus_batch(arr, a.ctx, a.eval_bs)
    t0 = time.time()
    for i in range(1, a.steps + 1):
        n_pl = max(1, a.bs // 4)
        n_rec = max(1, a.bs // 2)
        xr, yr, mr, rr, ar, gr = _dialogue_batch(rec_tr, a.ctx, n_rec, align=True)
        l_rec, _ = lm_loss(model(xr), yr, mr, xr, align=ar, gt_gate=gr)
        total = l_rec
        if a.mix:
            xp, yp, mp, rp = _dialogue_batch(pl_tr, a.ctx, n_pl)
            l_pl, _ = lm_loss(model(xp), yp, mp)
            total = total + l_pl
        xc, yc, mc, rc = _corpus_batch(arr, a.ctx, a.bs - n_rec - n_pl)
        l_co, _ = lm_loss(model(xc), yc, mc)
        total = total + l_co
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i % 300 == 0 or i == a.steps:
            model.eval()
            m = metrics(model, rec_b, True)
            ce_p = plain_ce(model, pl_b)
            ce_c = plain_ce(model, co_b)
            print(f"[{i:04d}/{a.steps}] rec_span={m['span']:.3f} "
                  f"rec_free={m['free']:.3f} ptr={m['ptr']:.3f} "
                  f"g_span={m['g_span']:.3f} plain_ce={ce_p:.3f} "
                  f"corpus_ce={ce_c:.3f} ({time.time() - t0:.0f}s)")
            model.train()

    model.eval()
    print("[chat] 普通模式（无检索）贪心：")
    for q in PROMPTS:
        ids = [BOS, U, *vocab.encode(q), A]
        out = greedy(model, ids, False, a.ctx, max_new=20)
        first = "<a>自环" if out and out[0] == A else "正常"
        print(f"  你: {q}")
        print(f"  它: {vocab.decode(out)}   [{first}]")
    print("[rec] 重构样例（开复制）：")
    for s in rec_te[:3]:
        out = greedy(model, [int(t) for t in s["x"][:max(
            j for j, t in enumerate(s["x"].tolist()) if t == A) + 1]],
            True, a.ctx, max_new=40)
        gen = vocab.decode(out).strip()
        gold = s["gold"]
        print(f"  真答: {gold}")
        print(f"  它:   {gen}  (cover={lcs_len(gen, gold) / max(len(gold), 1):.2f})")


if __name__ == "__main__":
    main()
