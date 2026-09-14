# 完整 Train800 LoRA SFT：执行配置与预算

2026-09-14：PR #100 已合并，commit `3ce2cc4aec03376ce31018edd4eba54b93a1c68b`。
用户授权：训练、验证和最终 Test100 的总开销低于 $5 时直接运行，并记录训练指标及配置。
用户已明确选择完整 RD Train800，peak LR 为 **1e-4**，保留 warmup。
已完成的两个 smoke 都仅使用合成 Train8，不能称为全量训练。

## RD Train800 的可执行预算方案

保持冻结 Train800 / Dev100 / Test100；不重新划分，不使用 Dev/Test 做梯度更新。
原始 RD 标签检查为 798/800 通过，另外 2 条含表格外文字和外部样式。已生成
仅保留完整表格的派生训练标签：800 条全部通过，图片、样本顺序及表格单元格/跨度
均保留；798 条标签不变。原文件、转换审计及哈希只保留本地。Dev/Test 标签不变。

| 设置 | 值 |
| --- | --- |
| 模型和初始化 | Qwen3.5-4B，新的 LoRA，不续训 smoke adapter |
| LoRA | rank 8，attention + MLP；不训练 unembed；alpha/dropout 未由接口暴露 |
| 训练量 | 完整 800 条，1 epoch，batch 8，100 optimizer steps |
| 优化器 | Adam，peak LR 1e-4，beta1 0.9 / beta2 0.95，eps 1e-8，weight decay 0，grad clip 1 |
| LR | 10 步线性 warmup，然后保持 peak LR |
| 序列 | 最大 16,384 tokens；不截断目标 HTML；只对 assistant 和结束 token 计算 loss |
| 梯度归一化 | 每个 batch 的监督 token mean |
| Train 监控 | 每步 batch NLL/PPL；完整 Train NLL/PPL 只在训练前后计算 |
| Dev 监控 | 完整 Dev100 NLL/PPL 在 0/25/50/75/100；完整生成在 0/100 |
| Test | 最终 checkpoint 对同一个 Test100 计算 NLL/PPL 和自由生成；不重复生成 Base |
| 生成协议 | greedy、8,192 output tokens、1,048,576 pixels；Test seed 20260913 |
| 保存 | 初始/最终 sampler、最终训练状态，TTL 7 天；最终 step 用于比较 |

学习率为第 1 步 1e-5、第 10 步 1e-4，此后保持 1e-4。
[Tinker 的 LoRA 指南](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/)
给出相对 full fine-tuning 约 10 倍的经验倍率；这不是对当前数据最优学习率的保证。
Dev 自由生成并行度为 4；周期检查全部包含 100 条，不随机缩小 Dev。

派生标签后重新核算：Train 输入 1,118,584；Dev 输入 148,946；Test 输入
172,332。最长完整序列不超过 9,516 / 9,270 / 12,455，因此使用 16,384
训练序列上限。以下是执行前预算，不是账单。

| 部分 | 计算费用上界 |
| --- | ---: |
| 完整 Train800，1 epoch | $0.824396 |
| Train 首尾 + Dev 五次 NLL | $0.984026 |
| Dev 首尾自由生成，均按达到输出上限估算 | $1.687130 |
| Test100 仅最终模型 NLL + 自由生成 | $0.901375 |
| 计算合计 | **$4.396927** |
| 加 10% 计算余量和 $0.10 存储预留 | **$4.936620** |

两 epochs、增加一次完整 Dev 自由生成，或每个 Dev 检查点都计算 Train800 NLL，
对应方案均超出 $5 计算上界，因此不在本轮预算内。存储预留基于 rank8 参数规模
保守估算；并非服务端实际文件大小或最终账单。费用核验依据为
[Tinker 官方价格](https://tinker-docs.thinkingmachines.ai/tinker/models/)：每百万
train $0.737、forward $0.33、sample $1.005；存储 $0.10/GB/月。没有预先扣除缓存折扣。

## 记录与比较

训练的 `--wandb-mode online` 使用隔离子进程实时记录 batch 曲线和完整 Dev 检查点，
同时保留本地逐 token 记录。W&B config 保存模型、初始化、LoRA、Adam 参数、epochs、
batch、seed、warmup、loss reduction、数据清单哈希、序列/图片/生成限制、评测频次、
SDK/renderer/评分器版本、代码哈希和预算。不会自动上传图片、HTML、样本 ID、真实路径、
checkpoint 地址或完整命令行。

最终评测使用 `checkpoint_eval --split test --stage after`；上传新结果时关联
[固定 Base Test100](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/9409ea84393aebb0)，
并在评测 run 中附上训练 config 和 training run ID。Base 不覆盖，比较协议不改变。

执行前检查：**223 passed / 3 skipped**，包括真实 W&B SDK 离线流式隐私检查、
标签派生、并发生成与完整 Dev 周期评估。训练与最终 Test100 已完成；
[实测结果](RD_TRAIN800_SFT_RESULTS.md)记录完整对照、费用与 W&B 读回验证。
训练与 Dev 的执行上限为 $3.50，Test 为 $0.91；加相同余量与存储预留为 $4.951。
费用门槛是 token 估算保护，不是供应商账单的硬限额。

## 复现入口

从仓库根目录运行。所有变量均由调用者提供真实路径；`TRAIN_MANIFEST` 指原始
Train800，`DEV_MANIFEST` / `TEST_MANIFEST` 指冻结清单；`DATA_ROOT` 必须能解析
清单里的相对路径。`LABEL_OUTPUT`、`TRAIN_OUTPUT` 和 `TEST_OUTPUT` 使用新的空目录。
派生标签目录必须位于 `DATA_ROOT` 内。

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.prepare_training_labels \
  --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT" --output-dir "$LABEL_OUTPUT"

uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.sft \
  --train-manifest "$LABEL_OUTPUT/rd_train.jsonl" --dev-manifest "$DEV_MANIFEST" \
  --data-root "$DATA_ROOT" --output-dir "$TRAIN_OUTPUT" \
  --tinker-cookbook-dir "$TINKER_COOKBOOK_DIR" --official-repo "$RD_OFFICIAL_REPO" \
  --epochs 1 --batch-size 8 --rank 8 --learning-rate 1e-4 --warmup-ratio 0.1 \
  --eval-every 25 --generate-every 100 --no-generate-train --train-nll-endpoints-only \
  --max-sequence-tokens 16384 --max-new-tokens 8192 \
  --max-train-examples 800 --max-dev-examples 100 --max-estimated-usd 3.50 \
  --dataset-label rd-train800-v1 --wandb-mode online --inference-concurrency 4 \
  --env-file "$ENV_FILE" --execute

uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.checkpoint_eval \
  --source-run "$TRAIN_OUTPUT/run.json" --manifest "$TEST_MANIFEST" \
  --data-root "$DATA_ROOT" --output-dir "$TEST_OUTPUT" \
  --tinker-cookbook-dir "$TINKER_COOKBOOK_DIR" --official-repo "$RD_OFFICIAL_REPO" \
  --split test --stage after --max-estimated-usd 0.91 --env-file "$ENV_FILE" --execute

uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.log_checkpoint_eval \
  --evaluation-dir "$TEST_OUTPUT" --manifest "$TEST_MANIFEST" --data-root "$DATA_ROOT" \
  --official-repo "$RD_OFFICIAL_REPO" --stage after --baseline-run-id 9409ea84393aebb0 \
  --env-file "$ENV_FILE" --upload
```

训练和评测去掉 `--execute` 只做本地校验/预算；发布去掉 `--upload` 只重评分并
生成可检查的汇总 payload。预检也会写输出目录，实际执行应选择另一个新目录。
Test 必须等训练进程及 W&B 收尾完成后开始，使来源 `run.json` 哈希保持稳定。
上述命令重跑会产生新的付费任务；历史结果比较优先复用已有预测和 checkpoint。
