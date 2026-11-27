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

也支持 SFT 的 `{"conversations": [{"role": ..., "content": ...}]}`（或 `from`/`value`）
格式，直接把这类文件指给 `E7_MYQA` / `E7_CORPUS` 即可，语料取全部轮次、问答取
相邻的 user→assistant 轮：

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

## 放大模型

`--big` 固定 d384/6 层/6 头（15.4M）。用 `--d` `--n-layer` `--n-head` 覆盖即可放大
（ctx 保持 192，样本不用重建；尺寸变了就不能 `--resume`）：

| d / L / H | 参数 | 权重+AdamW |
|---|---|---|
| 384 / 6 / 6（`--big`） | 15.4M | 0.19 GB |
| 512 / 8 / 8 | 32.6M | 0.39 GB |
| 640 / 10 / 10 | 59.8M | 0.72 GB |
| 768 / 12 / 12 | 99.3M | 1.19 GB |

```bash
python -m blm.run train --stage lm --d 512 --n-layer 8 --n-head 8 \
    --steps 20000 --bs 16 --lr 3e-4 --save-every 1000
```

数据量提醒：语料只有 1 亿字（`--max-chars 100000000`）时，按 20 token/参数 的粗略
口径，32M 参数需要 ~6.5 亿 token ≈ 21 万步（bs16/ctx192）；再大就会欠拟合数据、
先过拟合那 1.7 万条重构样本。当前数据下建议不超过 32M。

## 切换词表（字符级 / BPE）

默认字符级（1 token = 1 字）。加 `--bpe 32000` 改为 ByteLevel BPE，special 固定占
id 0–5（`<pad> <unk> <bos> <eos> <u> <a>`），其余 token 从 6 开始。

```bash
# 只训分词器（CPU，写 artifacts/tokenizer.json + vocab.json）
python -m blm.run prep --corpus-only --bpe 32000

# 用 BPE 重建全部数据（vocab.json 会带 "kind": "bpe"）
python -m blm.run prep --bpe 32000 --max-chars 490000000
python -m blm.run prep --dialogues-only --memory
python -m blm.run prep --dialogues-only --reconstruct
```

要点：

- `vocab.json` 的 `kind` 决定加载实现，训练/推理/评估自动识别，不用改命令；
- 字符级和 BPE 的产物互不兼容，换词表必须整套重建，且不能 `--resume` 旧 ckpt；
- 用 `E7_ARTIFACTS` 把两套实验分目录，避免互相覆盖；
- BPE 下 `corpus.npy` 存 uint16（vocab < 65536），体积减半；
- 重构样本的 span 对齐已按 `encode_offsets` 做字符→token 映射（单测 9/9 命中），
  不用再假设"字符下标 = token 下标"。

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
