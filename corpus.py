# -*- coding: utf-8 -*-
"""e7 数据管线：语料字符流 + 对话样本。"""
from __future__ import annotations

import collections
import difflib
import gzip
import json
import os
import random

import numpy as np

from .vocab import A, BOS, EOS, U, Vocab

# 路径可用环境变量覆盖（换机器时不用改代码）：E7_CORPUS / E7_DIALOGUE
CORPUS_SRC = os.environ.get(
    "E7_CORPUS", r"e:\workspace\yac\app\llm\pretrain_t2t_mini.jsonl")
DIALOGUE_SRC = os.environ.get(
    "E7_DIALOGUE", r"e:\workspace\yac\app\llm\chat_data.jsonl")
HERE = os.path.dirname(os.path.abspath(__file__))
LCCC_DIR = os.path.join(HERE, "data", "lccc")
LCCC_FILES = ("lccc_base_valid.jsonl.gz", "lccc_base_test.jsonl.gz")
LCCC_MAX = 20_000
# 过滤口径（§14 下一步）：1–4 轮、回复 5–50 字、去复读、去广告、精确去重。
LCCC_MIN_UTT = 2
LCCC_MAX_UTT = 8
LCCC_MIN_REPLY = 5
LCCC_MAX_REPLY = 50
LCCC_AD = ("公众号", "加我", "私聊", "扫码", "代购", "薇信", "威信", "http",
           "www", "兼职", "刷单", "贷款", "加微")
# A 方案：真实问答对话（question/answer），长回复、有信息量，替代 LCCC。
MYQA_SRC = os.environ.get("E7_MYQA", os.path.join(HERE, "data", "my.jsonl"))
MYQA_MAX = 50_000


SFT_KEYS = ("conversations", "messages", "conversation")
_ROLE_MAP = {"assistant": "assistant", "gpt": "assistant", "bot": "assistant",
             "ai": "assistant", "system": "user"}


def _turns_of(obj) -> list[tuple[str, str]]:
    """把 SFT 记录（`conversations` / `messages`）归一成 [(role, content)]。

    兼容 `{"role","content"}`、`{"from","value"}` 以及 content 为分段列表的形式。
    """
    raw = obj
    if isinstance(obj, dict):
        for k in SFT_KEYS:
            if k in obj:
                raw = obj[k]
                break
    if not isinstance(raw, list):
        return []
    out: list[tuple[str, str]] = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        role = str(t.get("role") or t.get("from") or "").lower()
        content = t.get("content", t.get("value", ""))
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p)
                              for p in content)
        content = str(content).strip()
        if not content:
            continue
        out.append((_ROLE_MAP.get(role, "user"), content))
    return out


def iter_texts(path: str, max_chars: int | None = None):
    """流式读语料：`{"text": ...}` 或 SFT 的 `{"conversations": [...]}`。"""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                t = obj.get("text") or "\n".join(c for _, c in _turns_of(obj))
            else:
                t = ""
            if not t:
                continue
            yield t
            n += len(t)
            if max_chars and n >= max_chars:
                return


def build_corpus(vocab: Vocab, path: str = CORPUS_SRC,
                 max_chars: int = 2_000_000) -> np.ndarray:
    ids: list[int] = []
    for t in iter_texts(path, max_chars):
        ids.extend(vocab.encode(t))
        if len(ids) >= max_chars:
            break
    return np.asarray(ids[:max_chars], dtype=np.int32)


def _read_conversations(path: str) -> list[list[tuple[str, str]]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msgs = json.loads(line).get("conversations") or []
            except json.JSONDecodeError:
                continue
            turns = [(m.get("role", "user"), m.get("content", "").strip())
                     for m in msgs if m.get("content")]
            if turns:
                out.append(turns)
    return out


def _is_repeat_reply(r: str) -> bool:
    """回复里单一字符占比 > 0.5 视为复读（如"哈哈哈哈哈哈"）。"""
    if len(r) < LCCC_MIN_REPLY:
        return False
    top = collections.Counter(r).most_common(1)[0][1]
    return top / len(r) > 0.5


def _lccc_keep(utts: list[str]) -> bool:
    if not (LCCC_MIN_UTT <= len(utts) <= LCCC_MAX_UTT):
        return False
    for i in range(1, len(utts), 2):
        r = utts[i]
        if not (LCCC_MIN_REPLY <= len(r) <= LCCC_MAX_REPLY):
            return False
        if _is_repeat_reply(r):
            return False
    joined = "".join(utts)
    return not any(k in joined for k in LCCC_AD)


def load_lccc(max_dialogues: int = LCCC_MAX,
              dirpath: str = LCCC_DIR) -> list[list[tuple[str, str]]]:
    """LCCC 真实多轮对话（已在仓库外预处理为逐字空格分隔，这里去掉空格）。

    每个文件每行是一个 utterance 列表，用户/助手交替。按 §14 口径过滤：
    1–4 轮、回复 5–50 字、去复读、去广告、精确去重；`max_dialogues` 计过滤后保留数。
    """
    out: list[list[tuple[str, str]]] = []
    seen: set[tuple[str, ...]] = set()
    for name in LCCC_FILES:
        path = os.path.join(dirpath, name)
        if not os.path.exists(path):
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    utts = [str(t).replace(" ", "") for t in json.loads(line)]
                except json.JSONDecodeError:
                    continue
                utts = [t for t in utts if t]
                if len(utts) < 2:
                    continue
                if len(utts) % 2:          # 末尾只有用户发言，丢掉
                    utts = utts[:-1]
                if not _lccc_keep(utts):
                    continue
                key = tuple(utts)
                if key in seen:
                    continue
                seen.add(key)
                out.append([("user" if i % 2 == 0 else "assistant", t)
                            for i, t in enumerate(utts)])
                if len(out) >= max_dialogues:
                    return out
    return out


def load_myqa(max_dialogues: int = MYQA_MAX, min_reply: int = 5,
              max_reply: int = 300, min_q: int = 2,
              seed: int = 0,
              path: str | None = None) -> list[list[tuple[str, str]]]:
    """真实问答对话（`my.jsonl` 的 question/answer）。

    过滤：问题 2 字以上、回复 5–300 字、去广告、精确去重；用 reservoir
    采样保留 `max_dialogues` 条以保证多样性。每条作为一轮 user/assistant。
    """
    rng = random.Random(seed)
    kept: list[list[tuple[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    path = path or MYQA_SRC
    if not os.path.exists(path):
        return kept
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = str(obj.get("question", "")).strip()
            a = str(obj.get("answer", "")).strip()
            if not (min_q <= len(q) and min_reply <= len(a) <= max_reply):
                continue
            if any(k in q or k in a for k in LCCC_AD):
                continue
            key = (q, a)
            if key in seen:
                continue
            seen.add(key)
            turns = [("user", q), ("assistant", a)]
            if len(kept) < max_dialogues:
                kept.append(turns)
            else:
                j = rng.randint(0, len(seen) - 1)
                if j < max_dialogues:
                    kept[j] = turns
    return kept


def _is_sft(path: str) -> bool:
    """看首条非空记录是不是 `conversations` 这类 SFT 结构。"""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                return isinstance(obj, dict) and any(k in obj for k in SFT_KEYS)
    except (OSError, json.JSONDecodeError):
        return False
    return False


def load_sft(path: str, max_dialogues: int = MYQA_MAX,
             max_turns: int = 8) -> list[list[tuple[str, str]]]:
    """SFT 多轮对话（moss / sft_t2t 的 `conversations`）。

    截前 `max_turns` 轮，丢掉含广告/网址的样本，`max_dialogues` 封顶。
    """
    out: list[list[tuple[str, str]]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                turns = _turns_of(json.loads(line))[:max_turns]
            except json.JSONDecodeError:
                continue
            if len(turns) < 2:
                continue
            if any(k in c for _, c in turns for k in LCCC_AD):
                continue
            out.append(turns)
            if len(out) >= max_dialogues:
                break
    return out


def load_conversations(path: str | None = None,
                       **_kw) -> list[list[tuple[str, str]]]:
    """对话数据源：自动识别 my.jsonl（question/answer）与 SFT（conversations）。"""
    path = path or MYQA_SRC
    if os.path.exists(path) and _is_sft(path):
        return load_sft(path)
    return load_myqa(path=path)


def build_dialogues(vocab: Vocab, ctx: int,
                    path: str | None = None) -> list[dict]:
    """每段对话 -> x / y / lm_mask / role，已按 ctx 截断。

    role: 0=用户, 1=助手, 2=收尾(EOS), -100=不计（<bos>）。
    lm_mask: 1=要预测的部分（助手内容 + 收尾），与 role 解耦——
             收尾既要计入 LM（学会停句），又是独立角色类别。
    """
    samples = []
    for turns in load_conversations(path):
        ids: list[int] = [BOS]
        role: list[int] = [-1]              # -1=bos, 0=用户, 1=助手, 2=收尾
        lm: list[float] = [0.0]             # 1=助手内容或收尾
        for role_name, content in turns:
            marker, sp = (U, 0) if role_name != "assistant" else (A, 1)
            for tok in [marker, *vocab.encode(content)]:
                ids.append(tok)
                role.append(sp)
                lm.append(1.0 if sp == 1 else 0.0)
            if sp == 1:
                ids.append(EOS)
                role.append(2)              # 收尾单独一类
                lm.append(1.0)              # 收尾也要学

        if len(ids) < 3:
            continue
        x = np.asarray(ids[:-1], dtype=np.int64)
        y = np.asarray(ids[1:], dtype=np.int64)
        lm_mask = np.asarray(lm[1:], dtype=np.float32)
        role_arr = np.asarray(role[1:], dtype=np.int64)

        if len(x) > ctx:
            x, y, lm_mask, role_arr = (a[-ctx:] for a in (x, y, lm_mask, role_arr))
        if lm_mask.sum() == 0:
            continue
        samples.append({"x": x, "y": y, "lm": lm_mask, "role": role_arr})
    return samples


def load_qa_pairs(path: str | None = None) -> list[tuple[str, str]]:
    """(question, answer) 对：多轮对话取相邻的 user→assistant 轮。"""
    out: list[tuple[str, str]] = []
    for turns in load_conversations(path):
        for i in range(len(turns) - 1):
            if turns[i][0] == "user" and turns[i + 1][0] == "assistant":
                out.append((turns[i][1], turns[i + 1][1]))
    return out


def build_memory_dialogues(vocab: Vocab, ctx: int, path: str | None = None,
                           max_answer: int = 60) -> list[dict]:
    """外部事实记忆样本：`<bos><u>Q<a>RET<eos><u>Q<a>GOLD<eos>`。

    RET 取样本自身答案（检索库含自身时 top1 即它，避免标签噪声），只当上下文，
    不计 LM/role；第二轮才是要学的 GOLD。教模型"在上下文里定位并抽取答案"，
    超 ctx 的样本丢弃（2*|Q| + |RET| + |GOLD| + 7 <= ctx）。
    内部语义记忆（`Memory`）照常由 L 训练；检索本身在推理时跑（`retrieval.py`）。
    """
    pairs = load_qa_pairs(path)
    samples = []
    for q, a in pairs:
        ret = a[:max_answer]
        ids: list[int] = [BOS]
        lm: list[float] = [0.0]
        role: list[int] = [-1]
        for tok in [U, *vocab.encode(q), A, *vocab.encode(ret), EOS]:
            ids.append(tok)
            lm.append(0.0)
            role.append(-1)
        for tok in [U, *vocab.encode(q)]:
            ids.append(tok)
            lm.append(0.0)
            role.append(0)
        for tok in [A, *vocab.encode(a)]:
            ids.append(tok)
            lm.append(1.0)
            role.append(1)
        ids.append(EOS)
        lm.append(1.0)
        role.append(2)

        if len(ids) - 1 > ctx:
            continue
        samples.append({
            "x": np.asarray(ids[:-1], dtype=np.int64),
            "y": np.asarray(ids[1:], dtype=np.int64),
            "lm": np.asarray(lm[1:], dtype=np.float32),
            "role": np.asarray(role[1:], dtype=np.int64),
        })
    return samples


def _bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _dice(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def answer_align(src: str, tgt: str,
                 min_block: int = 3) -> list[tuple[int, int, int]]:
    """真实答案对齐：返回 [(src_i, tgt_j, size)]，只保留 ≥min_block 的连续块。

    difflib 最长公共子序列块，不造任何改写：块内是"该逐字落地"的片段
    （专名/数字/术语/固定搭配），块外就是必须重新组织的部分。
    """
    sm = difflib.SequenceMatcher(None, src, tgt, autojunk=False)
    return [(b.a, b.b, b.size) for b in sm.get_matching_blocks()
            if b.size >= min_block]


def mine_reconstruct_pairs(pairs: list[tuple[str, str]],
                           min_dice: float = 0.45, max_dice: float = 0.75,
                           min_keep: float = 0.2, max_keep: float = 0.5,
                           min_block: int = 3, min_len: int = 8,
                           max_len: int = 60, max_pairs: int = 20000
                           ) -> list[dict]:
    """按答案相似度挖真实"近似问法"配对，供重构训练。

    候选由答案字符 bigram 倒排生成，再用 dice 对齐措辞相近但不相同的答案；
    `min_keep/max_keep` 限制"逐字块覆盖率"——太低没得抄（退化成生成），
    太高等于整句照抄（正是 §20.1.1 要禁的）。所有配对都是真实 Q/A。
    """
    uniq: dict[str, tuple[str, str]] = {}
    for q, a in pairs:
        if min_len <= len(a) <= max_len and a not in uniq:
            uniq[a] = (q, a)
    answers = list(uniq.values())
    grams = [_bigrams(a) for _, a in answers]
    index: dict[str, list[int]] = collections.defaultdict(list)
    for i, gs in enumerate(grams):
        for g in gs:
            index[g].append(i)

    out: list[dict] = []
    for i, (q1, a1) in enumerate(answers):
        if len(out) >= max_pairs:
            break
        cand: collections.Counter[int] = collections.Counter()
        # 只用 df 小的 bigram 生成候选（高频 gram 没有区分度、还会拖慢）
        for g in sorted(grams[i], key=lambda x: len(index[x]))[:6]:
            for j in index[g]:
                if j != i:
                    cand[j] += 1
        best = None
        for j, _ in cand.most_common(20):
            q2, a2 = answers[j]
            if a1 == a2:
                continue
            dice = _dice(grams[i], grams[j])
            if not (min_dice <= dice <= max_dice):
                continue
            blocks = answer_align(a1, a2, min_block)
            if not blocks:
                continue
            keep = sum(b[2] for b in blocks) / len(a2)
            if not (min_keep <= keep <= max_keep):
                continue
            # 问题要真的是不同问法，否则等于自检索
            if q1 == q2 or _dice(_bigrams(q1), _bigrams(q2)) > 0.85:
                continue
            score = abs(dice - (min_dice + max_dice) / 2) - 0.3 * keep
            if best is None or score < best[0]:
                best = (score, q2, a2, keep)
        if best is None:
            continue
        _, q2, a2, keep = best
        out.append({"q1": q1, "a1": a1, "q2": q2, "a2": a2, "keep": keep})
        # 反向也是真实配对：同一个答案相似度，换个方向问
        blocks = answer_align(a2, a1, min_block)
        if blocks:
            out.append({"q1": q2, "a1": a2, "q2": q1, "a2": a1,
                        "keep": sum(b[2] for b in blocks) / len(a1)})
    return out


def _rec_sample(vocab: Vocab, ctx: int, q1: str, a1: str, q2: str, a2: str,
                min_block: int = 3) -> dict | None:
    """`<bos><u>Q1<a>A1<eos><u>Q2<a>A2<eos>` + 目标位置→源位置的对齐标注。"""
    q1_ids, q2_ids = vocab.encode(q1), vocab.encode(q2)
    a1_ids, a2_ids = vocab.encode(a1), vocab.encode(a2)
    ids: list[int] = [BOS]
    lm: list[float] = [0.0]
    role: list[int] = [-1]
    for tok in [U, *q1_ids, A, *a1_ids, EOS]:
        ids.append(tok)
        lm.append(0.0)
        role.append(-1)
    src_start = 3 + len(q1_ids)                 # 第一轮答案首字在 ids 里的下标
    for tok in [U, *q2_ids]:
        ids.append(tok)
        lm.append(0.0)
        role.append(0)
    tgt_start = len(ids) + 1                    # <a> 之后的第一个答案字
    for tok in [A, *a2_ids]:
        ids.append(tok)
        lm.append(1.0)
        role.append(1)
    ids.append(EOS)
    lm.append(1.0)
    role.append(2)
    if len(ids) - 1 > ctx:
        return None

    n = len(ids) - 1
    align = np.full(n, -100, dtype=np.int64)    # x 下标 → 源字符 x 下标
    gtg = np.full(n, -100.0, dtype=np.float32)  # 门控目标：1=该抄的片段
    for si, ti, size in answer_align(a1, a2, min_block):
        for d in range(size):
            if ti + d >= len(a2_ids):
                continue
            p = tgt_start + ti + d - 1
            align[p] = src_start + si + d
            gtg[p] = 1.0
    for p in range(n):
        if lm[p + 1] > 0 and gtg[p] < 0:
            gtg[p] = 0.0                         # 目标里其余位置不该逐字抄
    return {
        "x": np.asarray(ids[:-1], dtype=np.int64),
        "y": np.asarray(ids[1:], dtype=np.int64),
        "lm": np.asarray(lm[1:], dtype=np.float32),
        "role": np.asarray(role[1:], dtype=np.int64),
        "align": align,
        "gt_gate": gtg,
        "src": a1,
        "gold": a2,
        "q": q2,
    }


def build_reconstruct_dialogues(vocab: Vocab, ctx: int,
                                path: str | None = None,
                                max_ret: int = 60, max_pairs: int = 20000,
                                min_keep: float = 0.2,
                                max_keep: float = 0.5) -> list[dict]:
    """重构样本：检索轮给真实近似问答，目标轮要求按当前问法重排答案。

    与 `build_memory_dialogues` 的区别：那里上下文就是标准答案（抄即最优），
    这里上下文答案与目标答案只共享"该逐字落地"的片段，其余必须改写。
    """
    items = mine_reconstruct_pairs(load_qa_pairs(path), max_pairs=max_pairs,
                                   min_keep=min_keep, max_keep=max_keep)
    samples = []
    for it in items:
        s = _rec_sample(vocab, ctx, it["q1"], it["a1"][:max_ret],
                        it["q2"], it["a2"][:max_ret])
        if s is not None:
            samples.append(s)
    return samples


def load_intents(vocab: Vocab,
                 path: str | None = None) -> tuple[list[dict], list[str]]:
    path = path or os.path.join(HERE, "data", "intent.jsonl")
    if not os.path.exists(path):
        return [], []
    labels: list[str] = []
    rows: list[tuple[list[int], str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text, label = obj.get("text", ""), obj.get("label", "")
            if not text or not label:
                continue
            if label not in labels:
                labels.append(label)
            rows.append((vocab.encode(text), label))
    return [{"ids": ids, "label": labels.index(label)} for ids, label in rows], labels
