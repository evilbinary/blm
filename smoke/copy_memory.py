# -*- coding: utf-8 -*-
"""表达冒烟：CopyHead + 门控（--mode copy）对比纯 LM（--mode lm）。

任务：`<bos><u>Q<a>RET<eos><u>Q<a>GOLD<eos>`，只学第二轮 GOLD；
数据来自 `artifacts/dialogues_mem.pt`，先跑
`python -m e7.run prep --dialogues-only --memory`。

指标：留出问法上的逐字 token/首字准确率、门控均值、贪心逐字命中。
冒烟只训表达损失（不含 role/conf/mem 辅助项），用来隔离复制机制。

python -m e7.smoke.copy_memory --mode lm   --steps 3000
python -m e7.smoke.copy_memory --mode copy --steps 3000
"""
from __future__ import annotations

import argparse
import os
import random
import time

import torch

from ..net.brain import BrainNet, Config, lm_loss
from ..net.train import _dialogue_batch
from ..vocab import A, EOS, Vocab

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts")


def mix_probs(out: dict, x: torch.Tensor) -> torch.Tensor:
    """把指针分布散回到词表，与生成头按门控混合。"""
    g = out["gate"].unsqueeze(-1)
    p = torch.softmax(out["logits"], -1)
    pc = torch.zeros_like(p)
    pc.scatter_add_(2, x.unsqueeze(1).expand(-1, out["copy_attn"].size(1), -1),
                    out["copy_attn"])
    return g * pc + (1.0 - g) * p


@torch.no_grad()
def acc_of(model: BrainNet, batch, use_copy: bool) -> tuple[float, float, float]:
    x, y, m, _ = batch
    out = model(x)
    pred = (mix_probs(out, x) if use_copy else out["logits"]).argmax(-1)
    valid = m > 0
    prev = torch.cat([torch.zeros_like(valid[:, :1]), valid[:, :-1]], 1)
    first = valid & ~prev
    gate = float(out["gate"][valid].mean()) if use_copy else 0.0
    return (float(((pred == y) & valid).sum()) / float(valid.sum()),
            float(((pred == y) & first).sum()) / float(first.sum()), gate)


def prompt_of(sample: dict) -> list[int]:
    ids = [int(t) for t in sample["x"]]
    return ids[:max(i for i, t in enumerate(ids) if t == A) + 1]


@torch.no_grad()
def greedy(model: BrainNet, prompt: list[int], use_copy: bool,
           ctx: int, max_new: int = 30) -> list[int]:
    ids = list(prompt)
    for _ in range(max_new):
        x = torch.tensor([ids[-ctx:]], dtype=torch.long)
        out = model(x)
        p = torch.softmax(out["logits"][0, -1], -1)
        if use_copy:
            gate = float(out["gate"][0, -1])
            pc = torch.zeros_like(p)
            pc.scatter_add_(0, x[0], out["copy_attn"][0, -1])
            p = gate * pc + (1.0 - gate) * p
        nxt = int(p.argmax())
        if nxt == EOS:
            break
        ids.append(nxt)
    return ids[len(prompt):]


def main() -> None:
    ap = argparse.ArgumentParser(description="e7 复制/表达冒烟")
    ap.add_argument("--mode", choices=["copy", "lm"], default="copy")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--ctx", type=int, default=192)
    ap.add_argument("--max-answer", type=int, default=24)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=600)
    ap.add_argument("--greedy", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    use_copy = a.mode == "copy"

    vocab = Vocab.load(os.path.join(ART, "vocab.json"))
    samples = torch.load(os.path.join(ART, "dialogues_mem.pt"),
                         weights_only=False)

    def gold_of(s: dict) -> str:
        return vocab.decode(s["y"][s["lm"] > 0])

    samples = [s for s in samples if 0 < len(gold_of(s)) <= a.max_answer]
    random.shuffle(samples)
    train_s = samples[:a.train]
    test_s = samples[a.train:a.train + a.test]

    cfg = Config(vocab_size=len(vocab), d=a.d, n_layer=a.n_layer,
                 n_head=a.n_head, ctx=a.ctx, dropout=a.dropout)
    model = BrainNet(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    print(f"[smoke] mode={a.mode} steps={a.steps} bs={a.bs} lr={a.lr:g} "
          f"d={a.d} L={a.n_layer} H={a.n_head} ctx={a.ctx} "
          f"max_ans={a.max_answer} seed={a.seed}")
    print(f"[smoke] params="
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"train={len(train_s)} test={len(test_s)}")

    fixed_train = _dialogue_batch(train_s, a.ctx, a.bs)
    fixed_test = _dialogue_batch(test_s, a.ctx, a.bs)
    t0 = time.time()
    run = None
    for i in range(1, a.steps + 1):
        x, y, m, _ = _dialogue_batch(train_s, a.ctx, a.bs)
        loss, _ = lm_loss(model(x), y, m, x if use_copy else None)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run = float(loss) if run is None else run * 0.9 + float(loss) * 0.1
        if i % 300 == 0 or i == a.steps:
            model.eval()
            tt = acc_of(model, fixed_train, use_copy)
            hh = acc_of(model, fixed_test, use_copy)
            g_t = f"{tt[2]:.3f}" if use_copy else "-"
            g_h = f"{hh[2]:.3f}" if use_copy else "-"
            print(f"[{i:04d}/{a.steps}] loss={run:.3f} "
                  f"train_tok={tt[0]:.3f} train_first={tt[1]:.3f} "
                  f"train_gate={g_t} held_tok={hh[0]:.3f} "
                  f"held_first={hh[1]:.3f} held_gate={g_h} "
                  f"({time.time() - t0:.0f}s)")
            model.train()

    model.eval()
    hits = 0
    n_eval = min(a.greedy, len(test_s))
    for s in test_s[:n_eval]:
        gold = gold_of(s)
        out = vocab.decode(greedy(model, prompt_of(s), use_copy, a.ctx)).strip()
        hits += int(out == gold)
        print(f"[{'hit' if out == gold else 'miss'}] 答: {gold}")
        print(f"       它: {out}")
    print(f"[smoke] mode={a.mode} 留出逐字命中 {hits}/{n_eval} "
          f"held_tok={hh[0]:.3f} held_first={hh[1]:.3f}")


if __name__ == "__main__":
    main()
