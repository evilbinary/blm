# -*- coding: utf-8 -*-
"""e7 功能模块：每个类对应 ARCHITECTURE.md §2 的一个公式。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Perception(nn.Module):
    """感知: h⁰ = E[x] + P[pos]"""

    def __init__(self, vocab_size: int, d: int, ctx: int, dropout: float = 0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(ctx, d)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(idx.size(1), device=idx.device)
        return self.drop(self.tok(idx) + self.pos(pos).unsqueeze(0))


class Block(nn.Module):
    """主干: causal self-attention + FFN，pre-LN。"""

    def __init__(self, d: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.nh = n_head
        self.hd = d // n_head
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(o))
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x


class Reference(nn.Module):
    """指代/省略: 每个位置对历史隐状态做注意力池化（因果，不看未来）。

    r_t = Σ_{i≤t} softmax(h_t·W_q·(h_i·W_k)/√d) · (h_i·W_v)
    早期实现对 `h[:, -1]` 做一次池化再广播到全序列，会把句尾信息泄漏到前面所有
    位置（教师强制准确率虚高、生成时对不上）；这里改成逐位置因果池化。
    """

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        t = h.size(1)
        scores = q @ k.transpose(1, 2) / math.sqrt(h.size(-1))
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool,
                                       device=h.device), diagonal=1)
        att = torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)
        return att @ v


class State(nn.Module):
    """对话状态: s_t = GRU(s_{t-1}, [h_t; r_t])"""

    def __init__(self, d: int):
        super().__init__()
        self.gru = nn.GRU(2 * d, d, batch_first=True)

    def forward(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if r.dim() == 2:                     # 兼容单个指代向量
            r = r.unsqueeze(1).expand(-1, h.size(1), -1)
        s, _ = self.gru(torch.cat([h, r], dim=-1))
        return s


class Memory(nn.Module):
    """语义记忆（学习型）: 事实槽上的软检索 m = Σ β_j v_j。

    与外部事实记忆（`retrieval.RetrievalMemory`）分工：
    这里不存具体事实，学的是"从状态里读出语义"的能力，K/V 由 L 更新；
    具体可写事实放在外部库，检索后作为上下文进序列。
    """

    def __init__(self, d: int, n_slots: int = 64):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.vals = nn.Parameter(torch.randn(n_slots, d) * 0.02)

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q: (B,d) 或 (B,T,d)。逐位置检索，让每个 token 都能读记忆。"""
        if q.dim() == 2:
            q = q.unsqueeze(1)
        beta = torch.softmax(q @ self.keys.t() / math.sqrt(q.size(-1)), dim=-1)
        return beta @ self.vals, beta


class CopyHead(nn.Module):
    """表达出口：字符级指针 + 复制门控。

    a_ti = softmax_i(h'_t·W_c·h_i/√d)，只允许看 i≤t（因果，不能抄下一个字）；
    p_copy(v) = Σ_{i:x_i=v} a_ti；
    g_t = σ(w_g·[h'_t; ctx_t])，ctx_t = Σ_i a_ti h_i。
    训练用混合概率的 CE，不需要对齐标注；门控零权重、bias=-4（起步≈纯生成）。
    """

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(2 * d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)

    def forward(self, hq: torch.Tensor,
                hk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hq: 表达侧 (B,T,d)（含 FiLM）；hk: 输入侧 (B,T,d)。"""
        q = self.q(hq)
        k = self.k(hk)
        t = q.size(1)
        scores = q @ k.transpose(1, 2) / math.sqrt(q.size(-1))
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool,
                                       device=q.device), diagonal=1)
        a = torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)
        ctx = a @ hk
        g = torch.sigmoid(self.gate(torch.cat([hq, ctx], dim=-1))).squeeze(-1)
        return a, g


class IntentHead(nn.Module):
    """意图/语用: u = softmax(W_u·h'_T)"""

    def __init__(self, d: int, n_intent: int):
        super().__init__()
        self.fc = nn.Linear(d, n_intent)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h[:, -1])


class PolicyHead(nn.Module):
    """策略: π_t = softmax(W_a·h'_t)，v0 预测轮次角色（用户/助手/收尾）。"""

    def __init__(self, d: int, n_role: int = 3):
        super().__init__()
        self.fc = nn.Linear(d, n_role)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class ConfidenceHead(nn.Module):
    """置信/元认知: c_t = σ(w·h'_t)"""

    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(h)).squeeze(-1)
