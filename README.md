# e7 — 字级对话模型（检索记忆 + 指针复制）

d384 / 6 层 / 6 头 / ctx192 的字级因果 Transformer（vocab ≈ 5943，约 15M 参数）。
内部语义记忆用槽位 `Memory` 训练；外部事实记忆用无参的字符 bigram 检索
（`net/retrieval.py`）；`CopyHead` 负责把上下文里"该逐字落地"的片段拷进回答，
门控决定哪里开、哪里关。

## 依赖

```bash
pip install torch numpy
```

Python ≥ 3.10，CPU 可跑（慢），GPU 自动使用。

## 数据

只需要两份原始数据，其余全部由 `prep` 生成：

| 文件 | 说明 |
|---|---|
| `data/my.jsonl` | 真实问答，字段 `question` / `answer`，约 27 万条 |
| `pretrain_t2t_mini.jsonl` | 预训练语料（用 `E7_CORPUS` 指定路径，默认是 Windows 本地路径） |

换机器不用改代码，直接给环境变量：

```bash
export E7_MYQA=/mnt/workspace/blm/data/my.jsonl
export E7_CORPUS=/mnt/workspace/data/pretrain_t2t_mini.jsonl
```

`data/`、`artifacts/` 不进仓库。

## 从头训练

```bash
# 1) 词表 + 语料 + 普通对话（默认只取 200 万字，要全量必须给 --max-chars）
python -m e7.run prep --max-chars 100000000

# 2) 派生样本（保持词表）
python -m e7.run prep --dialogues-only --memory        # 检索记忆样本
python -m e7.run prep --dialogues-only --reconstruct   # 重构样本（span 对齐）

# 3) 语料预热
python -m e7.run train --stage lm --big --steps 20000 --bs 16 --lr 3e-4 --save-every 1000

# 4) 联合训练：重构 1/2 + 普通对话 1/4 + 语料 1/4
python -m e7.run train --stage joint --reconstruct --big --resume \
    --steps 6000 --bs 16 --lr 5e-5 --save-every 500
```

注意：

- `--big` = d384/6 层/ctx192；`dialogues_mem.pt`、`dialogues_rec.pt` 按 ctx192 构建，
  `--memory` / `--reconstruct` 不能配 `--small`。
- 续训 lr 不要超过 5e-5：1.5e-4 实测会把 bpc 从 2.24 抬到 3.2。
- 产物约 1.5 GB（`corpus.npy` 400MB、`dialogues*.pt` ~400MB、`ckpt.pt` 194MB）。

## 推理

```bash
python -m e7.run chat                      # 无检索（当前版本此模式不可用，见下）
python -m e7.run chat --memory             # 先检索外部记忆再回答
python -m e7.run eval                      # 普通模式抽检
python -m e7.run eval --memory --n-mem 8   # 检索式评估：库内 + 库外
python -m e7.run info                      # 查看产物
```

## 冒烟工具（`e7/smoke/`）

```bash
python -m e7.smoke.retrieval_hit                       # 检索命中率
python -m e7.smoke.copy_memory --mode copy --steps 3000 # 指针/门控是否可学
python -m e7.smoke.reconstruct --mode copy --ckpt artifacts/ckpt.pt --greedy 12
python -m e7.smoke.mix_chat --mix 0|1 --steps 1200     # 混训对比（1.46M，CPU 可跑）
```

## 已知边界

- **无检索的 `chat` 不可用**：训练样本只包含「检索轮 + 问答轮」，`<u>问<a>` 前缀是
  分布外，模型会掉进 `<a>` 自环再退化到高频答案壳子。joint 阶段已混入普通对话来修，
  1.46M 验证：普通前缀 CE 5.55 → 4.55、自环消失。
- **检索没有 abstain**：字符 bigram 打分在库内/库外完全重叠（库外中位 0.40 反而高于
  库内非自身 0.34），阈值不可行；"不知道"只能作为训练信号引入。
- **多轮不支持**：`chat --memory` 每轮重置历史（对齐训练分布），追问拿不到上一轮信息。
- 置信头在因果修复前训练，当前偏乐观（错答也可能报 0.99）。

## 结构

```
e7/
  run.py               CLI: prep / train / chat / eval / info
  corpus.py            数据构建：语料、对话、检索记忆、重构配对
  vocab.py             字级词表
  net/brain.py         BrainNet、损失（lm / role / conf / intent）
  net/modules.py       注意力、槽位记忆、因果 Reference、CopyHead
  net/train.py         两段训练 + 混训采样
  net/sample.py        采样、检索推理、交互 chat
  net/retrieval.py     字符 bigram + IDF 的外部事实记忆
  smoke/               小模型冒烟与对比实验
```
