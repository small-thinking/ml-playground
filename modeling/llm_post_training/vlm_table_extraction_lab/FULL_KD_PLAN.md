# 全量 Train800 传统 KD

2026-09-14。用户要求删除 W&B 的 KD77 记录，执行全量 KD，与 Base / SFT800
在固定 Test100 上比较。旧 KD77 的训练和 Test 记录已删除并重新查询确认不存在；
本地原始实验与 teacher 缓存保留，用于复用和审计，不自动重新发布旧 run。

## 实验契约

使用与 SFT800 完全相同的800张训练图片、manifest顺序与训练shuffle seed。
Student从原始hosted Qwen3.5-4B新建LoRA，独立于之前的SFT/KD adapter。
Teacher冻结为Qwen3.6-35B-A3B；输入只有图片和既有prompt，不加入gold HTML。

| 项目 | 设置 |
| --- | --- |
| 训练数据 | 固定清洗版 Train800，800个唯一ID，800份可计算soft targets |
| Teacher rollout | greedy，关闭thinking，seed20260913，最多8192 tokens |
| 目标 | 原始teacher轨迹上的Top10 soft CE，温度1，Top10归一化 |
| 格式过滤 | 不过滤；原始token保持不变，不补EOS，不生成虚构续写 |
| 技术验证 | 非空token，已知stop状态，词表/TopK shape/概率合法，上下文不超16384 |
| Student | Qwen3.5-4B，fresh rank8 LoRA；attention/MLP开启，unembedding关闭 |
| 训练 | batch8，1 epoch，100 steps，seed20260914，与SFT800一致 |
| 优化器 | Adam，LR1e-4，warmup10，β=(0.9,0.95)，eps1e-8，clip1，weight decay0 |
| Dev | 完整100条；steps0/25/50/75/100 gold NLL/PPL；最终生成全量Dev |
| Test | 最终checkpoint评测固定Test100一次，复用已有Base/SFT800结果 |

之前KD77的3个排除是1个表格外/不支持的HTML内容和2个8192-token截断。
HTML parser通过是输出质量要求，并非token级KD的必要条件。本轮保留全部轨迹，
单独记录teacher格式失败/截断数量。缺失或损坏的技术数据仍使运行停止，不能静默
降成少于800条，也不能用gold答案修补teacher。Teacher错误可能被student学到，这是
本次监督来源的真实限制。

SFT800与KD800匹配图片、student初始化、LoRA、批次顺序、epoch、更新次数和LR。
两者监督来源和答案长度不同，因此仍需报告训练token数；不能称为等token/等成本对照。
若要隔离soft概率本身的收益，还需要同teacher轨迹的hard-target对照，未包含在本轮。

## 缓存与预算

新cache按ID、图片hash、prompt hash导入77份完整teacher缓存，另3份旧原始rollout
只补Top10概率；剩余720张各采集一次。不能直接复用索引：`sample(rows,80)`和
`sample(rows,800)`的顺序并不相同。旧cache保持不变，新cache保存来源及文件哈希。

Teacher采集并发4；每个请求提交前在同一个线程安全的持久化账本中保留费用，
总预算覆盖所有并发中的请求。失败取消未开始的任务；不确定请求保留pending状态，
禁止自动付费重试。已完成缓存可以复用。

按首轮token长度估算，本次新增约$5–6，预留约$7；格式失败/截断轨迹也会增加
训练长度，采集完成后必须用真实tokens重新计算训练预算。Teacher Dev和旧缓存
属于既有费用，不重复计入新增。所有金额是token估算，不是服务商账单硬限额。

## 复现入口

所有实际路径由调用者提供；data/outputs继续Git忽略，W&B仅上传汇总及允许的配置。

```bash
KD_PACKAGE=modeling.llm_post_training.vlm_table_extraction_lab
uv run --no-sync python -m "$KD_PACKAGE.kd_collect" \
  --train-manifest "$TRAIN_MANIFEST" --dev-manifest "$DEV_MANIFEST" \
  --data-root "$DATA_ROOT" --cache-dir "$KD800_CACHE" --reuse-cache "$KD80_CACHE" \
  --tinker-cookbook-dir "$COOKBOOK_DIR" --limit 800 --max-new-examples 800 \
  --include-invalid-rollouts --concurrency 4 --max-estimated-usd "$COLLECTION_BUDGET"
# 预检通过后追加 --execute --env-file "$ENV_FILE"。

uv run --no-sync python -m "$KD_PACKAGE.kd" \
  --cache-dir "$KD800_CACHE" --train-manifest "$TRAIN_MANIFEST" \
  --dev-manifest "$DEV_MANIFEST" --data-root "$DATA_ROOT" \
  --output-dir "$TRAIN_OUTPUT" --tinker-cookbook-dir "$COOKBOOK_DIR" \
  --official-repo "$RD_REPO" --train-examples 800 --include-invalid-rollouts \
  --batch-size 8 --epochs 1 --seed 20260914 --learning-rate 1e-4 \
  --warmup-ratio 0.1 --eval-every 25 --generate-dev \
  --wandb-mode online --dataset-label rd-kd-train800-v1 \
  --max-estimated-usd "$TRAINING_BUDGET"
# 预检与实际执行使用不同的全新output-dir。
```

训练完成后使用既有`checkpoint_eval`与`log_checkpoint_eval`，沿用同一固定Test分组。
