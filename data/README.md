# data 目录

## 数据文件说明

- `raw/` — 原始数据（MBPP JSON 等）
- `grpo_train.parquet` — veRL 格式训练数据（本地，MBPP train+val ~464条）
- `grpo_train_full.parquet` — veRL 格式扩充数据（云端，MBPP + APPS ~3000条）
- `grpo_multi_train.parquet` — 多轮训练数据（含 agent_name 列）
- `apps_cleaned.jsonl` — APPS 清洗后数据（由 scripts/synth_testcases.py 生成）

## 生成命令

```bash
# 本地基础数据（MBPP train+val ~464条）
python src/data_prepare.py --output data/grpo_train.parquet

# 云端扩充数据（MBPP + APPS ~3000条）
python src/data_prepare.py --output data/grpo_train_full.parquet --full

# 多轮训练数据
python src/data_prepare.py --output data/grpo_multi_train.parquet --multiturn
```
