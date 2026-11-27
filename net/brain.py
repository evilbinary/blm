# -*- coding: utf-8 -*-
"""e7 BrainNet：把功能模块串成一条可端到端训练的链。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    Block,
    ConfidenceHead,
    CopyHead,
    IntentHead,
    Memory,
    Perception,
    PolicyHead,
    Reference,
    State,
)


@dataclass
class Config:
    vocab_size: int = 6000
    d: int = 256
    n_layer: int = 4
    n_head: int = 4
    ctx: int = 192
    n_slots: int = 64
    n_intent: int = 6
    n_role: int = 3
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class BrainNet(nn.Module):
    """x → 感知 → 主干 → 指代 → 状态 → 记忆 → 表达 / 各功能头。"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.perception = Perception(cfg.vocab_size, cfg.d, cfg.ctx, cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg.d, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layer)])
        self.reference = Reference(cfg.d)
        self.state = State(cfg.d)
        self.memory = Memory(cfg.d, cfg.n_slots)
        self.copy_head = CopyHead(cfg.d)
        self.film_s = nn.Linear(cfg.d, cfg.d)
        self.film_m = nn.Linear(cfg.d, cfg.d)
        nn.init.zeros_(self.film_s.weight)
        nn.init.zeros_(self.film_s.bias)
        nn.init.zeros_(self.film_m.weight)
        nn.init.zeros_(self.film_m.bias)
        self.intent = IntentHead(cfg.d, cfg.n_intent)
        self.policy = PolicyHead(cfg.d, cfg.n_role)
        self.conf = ConfidenceHead(cfg.d)
        self.lm = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        self.lm.weight = self.perception.tok.weight      # 表达与感知共享词嵌入

    def forward(self, idx: torch.Tensor) -> dict:
        h = self.perception(idx)
        for blk in self.blocks:
            h = blk(h)
        r = self.reference(h)
        s = self.state(h, r)
        m, beta = self.memory(s)              # 逐位置读记忆 (B,T,d)
        h_ctx = h                             # 输入侧隐状态，供指针复制
        h = h + self.film_s(s) + self.film_m(m)
        copy_attn, gate = self.copy_head(h, h_ctx)
        return {
            "logits": self.lm(h),
            "role": self.policy(h),
            "conf": self.conf(h),
            "intent": self.intent(h),
            "mem": m,
            "mem_beta": beta,
            "copy_attn": copy_attn,
            "gate": gate,
        }


def lm_loss(out: dict, y: torch.Tensor, mask: torch.Tensor,
            x: torch.Tensor | None = None, align: torch.Tensor | None = None,
            gt_gate: torch.Tensor | None = None, w_align: float = 0.5,
            w_gate: float = 0.2, w_ent: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:
    """表达损失：生成头 + 复制指针的门控混合。

    p(y_t) = g_t·p_copy(y_t) + (1-g_t)·p_vocab(y_t)；
    训练只算目标 token 的混合概率（不展开 V 维复制分布），推理才展开。

    重构监督（`align`/`gt_gate` 非空时）：
    - `align[t]` = 目标字 t 在上下文里的真实源位置（LCS 块内），只监督这些位置的
      指针，逼"该抄的片段抄准"，其余位置不给指针监督；
    - `gt_gate[t]` = 1 该逐字落地 / 0 该改写，门控用 BCE 学选择性，避免整句常开；
    - 对齐位置的指针熵惩罚，让注意力峰值化，不把质量摊到高频字上。
    """
    logits = out["logits"]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                         y.reshape(-1), reduction="none").view_as(y)
    extra = torch.zeros((), device=logits.device)
    if x is not None and "copy_attn" in out:
        attn = out["copy_attn"]
        logp_gen = F.log_softmax(logits, dim=-1).gather(
            -1, y.unsqueeze(-1)).squeeze(-1)
        hit = ((x.unsqueeze(1) == y.unsqueeze(-1)).float()
               * attn).sum(-1).clamp_min(1e-8)
        g = out["gate"].clamp(1e-4, 1 - 1e-4)
        ce = -torch.logaddexp(torch.log(g) + torch.log(hit),
                              torch.log1p(-g) + logp_gen)
        if align is not None:
            m = align >= 0
            if m.any():
                idx = align.clamp(min=0).unsqueeze(-1)
                a_sel = attn.gather(-1, idx).squeeze(-1).clamp_min(1e-8)
                # 已对齐位置的指针 CE：直接指向真实对应字符
                ce = ce - (w_align * torch.log(a_sel) * m.float())
                ent = -(attn.clamp_min(1e-8).log() * attn).sum(-1)
                ce = ce + w_ent * ent * m.float()
        if gt_gate is not None:
            gm = gt_gate >= 0
            if gm.any():
                bce = F.binary_cross_entropy(g, gt_gate.clamp(0, 1),
                                             reduction="none")
                extra = extra + w_gate * (bce * gm.float()).sum() / gm.sum()
    return (ce * mask).sum() / mask.sum().clamp(min=1.0) + extra, ce


def load_state_compat(model: nn.Module, state: dict) -> list[str]:
    """加载旧 ckpt：新增的 copy_head 参数保留初始化值，其余必须严格匹配。"""
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith("copy_head.")]
    if bad or unexpected:
        raise RuntimeError(f"ckpt 与模型不匹配: missing={bad} "
                           f"unexpected={unexpected}")
    return missing


def role_loss(out: dict, role: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """轮次角色 CE + 真实类别指标（0=用户, 1=助手, 2=收尾）。

    返回 loss 与 {role_acc, role_f1(宏平均), role_r0/r1/r2(各类召回)}；
    无有效标签（如语料批）时返回空统计，避免把无标签批混进指标。
    """
    valid = role >= 0
    if not valid.any():
        return torch.zeros((), device=role.device), {}
    logits = out["role"]
    n_cls = logits.size(-1)
    tgt = role.clamp(min=0)
    ce = F.cross_entropy(logits.reshape(-1, n_cls), tgt.reshape(-1),
                         reduction="none").view_as(role)
    loss = (ce * valid).sum() / valid.sum()
    pred = logits.argmax(-1)
    correct = (pred == tgt) & valid
    stats = {"role_acc": float(correct.sum().float() / valid.sum())}
    f1s = []
    for c in range(n_cls):
        m = valid & (tgt == c)
        n_c = int(m.sum())
        if n_c == 0:
            stats[f"role_r{c}"] = 0.0
            continue
        recall = float((correct & m).sum()) / n_c
        prec_mask = (pred == c) & valid
        denom = int(prec_mask.sum())
        prec = float((correct & prec_mask).sum()) / denom if denom else 0.0
        f1 = 0.0 if prec + recall == 0 else 2 * prec * recall / (prec + recall)
        stats[f"role_r{c}"] = recall
        f1s.append(f1)
    stats["role_f1"] = float(sum(f1s) / len(f1s)) if f1s else 0.0
    return loss, stats


def conf_loss(out: dict, y: torch.Tensor, mask: torch.Tensor) -> tuple:
    """置信/元认知：直接学"这一步预测对不对"，并按 ECE 报告校准。

    target = 1[argmax(logits) == y]，置信头输出即 P(预测正确)。
    返回 loss 与 {conf_ece, conf_brier, conf_acc, conf_mean}。
    """
    with torch.no_grad():
        correct = (out["logits"].argmax(-1) == y).float()
    conf = out["conf"]
    bce = F.binary_cross_entropy(conf.clamp(1e-4, 1 - 1e-4), correct,
                                 reduction="none")
    loss = (bce * mask).sum() / mask.sum().clamp(min=1.0)

    valid = mask > 0
    c = conf[valid]
    ok = correct[valid]
    n = c.numel()
    if n == 0:
        return loss, {"conf_ece": 0.0, "conf_brier": 0.0,
                      "conf_acc": 0.0, "conf_mean": 0.0}
    bins = torch.clamp((c * 10).long(), 0, 9)          # 10 个等宽置信桶
    cd, okd = c.detach(), ok.detach()
    ece = 0.0
    for b in range(10):
        mb = bins == b
        cnt = int(mb.sum())
        if cnt:
            ece += (cnt / n) * abs(float(cd[mb].mean()) - float(okd[mb].mean()))
    return loss, {
        "conf_ece": float(ece),
        "conf_brier": float(((c - ok) ** 2).mean()),
        "conf_acc": float(ok.mean()),
        "conf_mean": float(c.mean()),
    }


def intent_loss(out: dict, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    l = F.cross_entropy(out["intent"], labels)
    acc = float((out["intent"].argmax(-1) == labels).float().mean())
    return l, acc
