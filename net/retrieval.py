# -*- coding: utf-8 -*-
"""e7 外部事实记忆：字符 bigram 倒排索引，无参数、确定性。

与内部语义记忆（`modules.Memory`，学习的 K/V）分工：具体、可写、可替换的事实
放这里；网络不把事实记进参数，只学着"读检索上下文、把答案落到输出"。
索引用 IDF 加权 + 余弦归一，避免长问题靠长度压过短问题。
"""
from __future__ import annotations

import math
from collections import defaultdict


class RetrievalMemory:
    """外部事实记忆库：可写（重建索引）、无梯度，检索结果进序列当上下文。"""

    def __init__(self, pairs, max_answer: int = 60):
        self.questions = [q for q, _ in pairs]
        self.answers = [a[:max_answer] for _, a in pairs]
        self.grams = [self._bigrams(q) for q in self.questions]
        self.index: dict[str, list[int]] = defaultdict(list)
        for i, gs in enumerate(self.grams):
            for g in gs:
                self.index[g].append(i)
        n = max(len(self.questions), 1)
        self.idf = {g: math.log((n + 1) / (len(ids) + 1)) + 1.0
                    for g, ids in self.index.items()}
        self.norm = [math.sqrt(sum(self.idf[g] for g in gs)) or 1.0
                     for gs in self.grams]

    def __len__(self) -> int:
        return len(self.questions)

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        text = text.strip()
        if len(text) < 2:
            return {text} if text else set()
        return {text[i:i + 2] for i in range(len(text) - 1)}

    def search(self, query: str, k: int = 1,
               max_grams: int = 8) -> list[tuple[float, str]]:
        """返回 [(得分, 答案)]，按得分降序；无任何 bigram 命中时返回 []。

        候选只由 df 最小的若干 bigram 生成（"什么/怎么"这类高频 gram 会拖慢检索，
        且几乎没有区分度），打分仍按 IDF 加权余弦归一。
        """
        qg = self._bigrams(query)
        grams = sorted((g for g in qg if g in self.index),
                       key=lambda g: len(self.index[g]))[:max_grams]
        if not grams:
            return []
        qnorm = math.sqrt(sum(self.idf.get(g, 0.0) for g in qg)) or 1.0
        scores: dict[int, float] = {}
        for g in grams:
            w = self.idf[g]
            for i in self.index[g]:
                scores[i] = scores.get(i, 0.0) + w
        ranked = sorted(scores.items(),
                        key=lambda x: (-x[1] / (qnorm * self.norm[x[0]]), x[0]))
        return [(s / (qnorm * self.norm[i]), self.answers[i])
                for i, s in ranked[:k]]
