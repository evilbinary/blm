# -*- coding: utf-8 -*-
"""重构冒烟：近似问法配对 + span 对齐监督（--mode copy）对比纯 LM。

任务：`<bos><u>Q1<a>A1<eos><u>Q2<a>A2<eos>`，检索轮是真实近似问答，
目标轮要按 Q2 重排 A2；A1/A2 只共享"该逐字落地"的片段（LCS 块），
其余位置必须改写。数据来自 `artifacts/dialogues_rec.pt`，先跑
`python -m e7.run prep --dialogues-only --reconstruct`。

指标（§20.1.1 口径）：整句逐字率只作参考，主看片段保留率（相对真答的覆盖）、
照抄来源比例（相对检索答案的覆盖，越低越像重构）、门控开在 span / 非 span 的比例、
指针在 span 上的命中率。

python -m e7.smoke.reconstruct --mode lm   --steps 3000
python -m e7.smoke.reconstruct --mode copy --steps 3000
"""
from __future__ import annotations

import argparse
import difflib
import os
import random
import time

import torch

from ..net.brain import BrainNet, Config, lm_loss, load_state_compat
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


def lcs_len(a: str, b: str) -> int:
    """最长公共子序列字符数（difflib 块求和，够用的覆盖度口径）。"""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(bl.size for bl in sm.get_matching_blocks())


def copy_span_len(gen: str, src: str, min_block: int = 3) -> int:
    """输出里逐字来自来源的字符数（只算 ≥3 字的连续块）。

    比整段 LCS 更贴近"片段复制"：中文常用字/标点满篇都是，整段 LCS 会把
    巧合重合算成照抄；块内连续 3 字以上才算真的把来源片段落地。
    """
    sm = difflib.SequenceMatcher(None, gen, src, autojunk=False)
    return sum(bl.size for bl in sm.get_matching_blocks() if bl.size >= min_block)


@torch.no_grad()
def metrics(model: BrainNet, batch, use_copy: bool) -> dict:
    dev = next(model.parameters()).device
    x, y, m, _, al, gg = (t.to(dev) for t in batch)
    out = model(x)
    pred = (mix_probs(out, x) if use_copy else out["logits"]).argmax(-1)
    valid = m > 0
    span = valid & (gg > 0.5)
    free = valid & (gg == 0)
    gate = out["gate"]
    hits = (out["copy_attn"].argmax(-1) == al) & span
    return {
        "tok": float(((pred == y) & valid).sum()) / max(float(valid.sum()), 1),
        "span": float(((pred == y) & span).sum()) / max(float(span.sum()), 1),
        "free": float(((pred == y) & free).sum()) / max(float(free.sum()), 1),
        "ptr": float(hits.sum()) / max(float(span.sum()), 1) if use_copy else 0.0,
        "g_span": float(gate[span].mean()) if span.any() else 0.0,
        "g_free": float(gate[free].mean()) if free.any() else 0.0,
    }


def prompt_of(sample: dict) -> list[int]:
    ids = [int(t) for t in sample["x"]]
    return ids[:max(i for i, t in enumerate(ids) if t == A) + 1]


@torch.no_grad()
def greedy(model: BrainNet, prompt: list[int], use_copy: bool, ctx: int,
           max_new: int = 40) -> list[int]:
    dev = next(model.parameters()).device
    ids = list(prompt)
    for _ in range(max_new):
        x = torch.tensor([ids[-ctx:]], dtype=torch.long, device=dev)
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
    ap = argparse.ArgumentParser(description="e7 重构冒烟")
    ap.add_argument("--mode", choices=["copy", "lm"], default="copy")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--ctx", type=int, default=192)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=800)
    ap.add_argument("--greedy", type=int, default=12)
    ap.add_argument("--eval-bs", type=int, default=32)
    ap.add_argument("--ckpt", default="",
                    help="直接评估已有 ckpt（跳过训练）")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    use_copy = a.mode == "copy"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab.load(os.path.join(ART, "vocab.json"))
    samples = torch.load(os.path.join(ART, "dialogues_rec.pt"),
                         weights_only=False)
    random.shuffle(samples)
    train_s = samples[:a.train]
    test_s = samples[a.train:a.train + a.test]

    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = Config(**ck["cfg"])
        model = BrainNet(cfg)
        load_state_compat(model, ck["state"])
        a.ctx = cfg.ctx
        steps = 0
        print(f"[smoke] eval ckpt={a.ckpt} step={ck.get('step')} "
              f"d={cfg.d} L={cfg.n_layer} ctx={cfg.ctx}")
    else:
        cfg = Config(vocab_size=len(vocab), d=a.d, n_layer=a.n_layer,
                     n_head=a.n_head, ctx=a.ctx, dropout=a.dropout)
        model = BrainNet(cfg)
        steps = a.steps
    model.to(dev)
    model.eval()                     # 评估 ckpt 时不能带 dropout，否则结果不可复现
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    print(f"[smoke] mode={a.mode} steps={steps} bs={a.bs} lr={a.lr:g} "
          f"d={cfg.d} L={cfg.n_layer} H={cfg.n_head} ctx={a.ctx} seed={a.seed} "
          f"device={dev.type}")
    print(f"[smoke] params="
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"train={len(train_s)} test={len(test_s)}")

    train_b = _dialogue_batch(train_s, a.ctx, min(a.eval_bs, len(train_s)),
                              align=True)
    test_b = _dialogue_batch(test_s, a.ctx, min(a.eval_bs, len(test_s)),
                             align=True)
    t0 = time.time()
    run = None
    for i in range(1, steps + 1):
        x, y, m, _, al, gg = (t.to(dev) for t in
                              _dialogue_batch(train_s, a.ctx, a.bs, align=True))
        out = model(x)
        if use_copy:
            loss, _ = lm_loss(out, y, m, x, align=al, gt_gate=gg)
        else:
            loss, _ = lm_loss(out, y, m)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run = float(loss) if run is None else run * 0.9 + float(loss) * 0.1
        if i % 300 == 0 or i == steps:
            model.eval()
            tt = metrics(model, train_b, use_copy)
            hh = metrics(model, test_b, use_copy)
            extra = (f"ptr={hh['ptr']:.3f} g_span={hh['g_span']:.3f} "
                     f"g_free={hh['g_free']:.3f}") if use_copy else ""
            print(f"[{i:04d}/{steps}] loss={run:.3f} "
                  f"train_tok={tt['tok']:.3f} held_tok={hh['tok']:.3f} "
                  f"held_span={hh['span']:.3f} held_free={hh['free']:.3f} "
                  f"{extra} ({time.time() - t0:.0f}s)")
            model.train()

    model.eval()
    if steps == 0:
        hh = metrics(model, test_b, use_copy)
        print(f"[eval] held_tok={hh['tok']:.3f} held_span={hh['span']:.3f} "
              f"held_free={hh['free']:.3f} ptr={hh['ptr']:.3f} "
              f"g_span={hh['g_span']:.3f} g_free={hh['g_free']:.3f}")
    n_eval = min(a.greedy, len(test_s))
    tot = {"exact": 0, "cover": 0.0, "copy": 0.0, "gate": 0.0, "verbatim": 0}
    for s in test_s[:n_eval]:
        gold, src = s["gold"], s["src"]
        gen = vocab.decode(greedy(model, prompt_of(s), use_copy, a.ctx)).strip()
        cover = lcs_len(gen, gold) / max(len(gold), 1)
        copy = copy_span_len(gen, src) / max(len(gen), 1)
        verbatim = int(bool(gen) and gen in src)
        with torch.no_grad():
            x1 = torch.tensor([s["x"].tolist()[-a.ctx:]], device=dev)
            gate = float(model(x1)["gate"][0].mean())
        exact = int(gen == gold)
        tot["exact"] += exact
        tot["cover"] += cover
        tot["copy"] += copy
        tot["gate"] += gate
        tot["verbatim"] += verbatim
        print(f"[{exact and 'exact' or 'gen  '}] 问: {s['q']}")
        print(f"         检索: {src}")
        print(f"         真答: {gold}")
        print(f"         它: {gen}")
        print(f"         cover={cover:.2f} 照抄块={copy:.2f} gate={gate:.2f}"
              f"{' 整句照抄' if verbatim else ''}")
    print(f"[smoke] mode={a.mode} exact={tot['exact']}/{n_eval} "
          f"cover={tot['cover']/n_eval:.3f} 照抄来源={tot['copy']/n_eval:.3f} "
          f"整句照抄={tot['verbatim']}/{n_eval} "
          f"门控均值={tot['gate']/n_eval:.3f}")


if __name__ == "__main__":
    main()
