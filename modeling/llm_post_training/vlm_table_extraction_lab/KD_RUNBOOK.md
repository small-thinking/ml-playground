# MoE teacher → dense student：第一次传统蒸馏

2026-09-14。已完成真实 teacher Top10 与 LoRA 反向传播烟测，计算费用估计 $0.140824。
8条采集中7条有效，实际训练7条、2步；完整记录见 [KD_SMOKE_RESULTS.md](KD_SMOKE_RESULTS.md)。
工程烟测单独限制在 $0.50 内。用户随后已授权正式 KD 训练和固定 Test100 评测，
采用预选80条候选及显式无效目标过滤；不从SFT800继续训练。
正式运行也已完成：77条有效目标、20步、完整Dev100与Test100，本次新增估算$1.203783。
三版本指标、训练曲线、费用和限制见 [RD_KD80_RESULTS.md](RD_KD80_RESULTS.md)。

本地验证：开启可选 pinned-renderer / official-scorer 集成后，lab 测试 **174 passed**，
Black、Ruff、`git diff --check` 通过。
测试包含概率与梯度、失败后付费重试保护、缓存篡改拒绝、完整 Dev100 的模拟训练流程和
离线 W&B 隐私检查。真实 Tinker Top10 反向传播另由上述付费烟测验证。

## 固定设置

| 项目 | 设置 |
| --- | --- |
| Teacher | 冻结 Qwen3.6-35B-A3B，关闭 thinking，greedy rollout |
| Student | 原始 Qwen3.5-4B，新建 LoRA；不加载 SFT800 adapter |
| 数据 | 冻结 Train800 中按 seed=20260913 选 80 条；烟测取该子集前 8 条 |
| 目标 | Teacher 自己生成的原始 token 轨迹上，Top10 概率 soft CE |
| 温度 | 分布 τ=1；rollout temperature=0，与 loss 温度分别记录 |
| LoRA | rank8；attention/MLP 开启，unembedding 关闭 |
| 优化器 | Adam，LR=1e-4，β=(0.9,0.95)，eps=1e-8，clip=1，weight decay=0 |
| 步数 | 烟测 7 条有效目标、2 步、warmup1；pilot 77/80 条有效目标、20 步、warmup2 |
| Dev | 全量100；初始、每10步、结束时 gold NLL/PPL；pilot结束生成完整Dev100 |
| Test | 最终 checkpoint 在固定 Test100 上评测一次，与已有 Base 和 SFT800 比较 |

Teacher Top10 先归一化，student 使用其完整词表 softmax 下的 log probability，
**不能在 student 的十个候选内再次归一化**。损失按 batch 的有效 assistant
位置总数平均；图像、prompt 和 padding 不参与 loss。
`soft_cross_entropy - teacher_entropy = truncated_forward_kl`；这不是完整 teacher 分布的 KL。
普通 perplexity 只报告 gold Dev NLL 的指数，不给 soft CE 的指数冠以同一名称。

## 代码阅读顺序

1. `kd_targets.py`：shift、二维 targets/weights、batch normalization、CE/entropy/KL。
2. `kd_collect.py`：固定采样、tokenizer/prompt一致性、两次 teacher 调用、缓存与预算。
3. `kd.py`：新建 LoRA、forward_backward / optim_step、Dev 检查和最终 sampler。

缓存的 `manifest.json` 固定模型/processor/cookbook/数据/采样参数和样本顺序。
每条保存原始 rollout token、完整原始 Top10 响应、完成段概率和 SHA-256。
不通过 decode→encode 重建训练目标，不静默丢弃坏格式或截断输出，不退化为 hard SFT。
Top10 mass 要求均值 ≥0.98 且 P05 ≥0.90，未达标时先停下分析。

默认仍在无效输出处停止。正式运行显式启用 `--skip-rejected`：只排除格式无效或
截断的teacher输出，保留原始token/HTML和带哈希的`.rejected.json`，不清洗、不重试、
不补换其他图片。采集和训练都重新验证排除原因；缺失概率、预算错误和不确定请求
不能借这个开关跳过。候选数、排除数、实际训练数与过滤策略分别记录到W&B。

调用前记录预算 reservation，成功且缓存落盘后才结算。未确定请求或持久化失败
会留下 `usage.json.pending` 并阻止自动重试，需要人工核对已有文件和计费；
普通完整缓存可以直接复用。训练中断不自动重新开始或恢复 optimizer，以免重复付费。
所有金额都是 token 单价估计，额外保留 10%；它们不是服务商账单的硬限额。

## 运行步骤

路径全部由调用者提供。下面变量代表本地配置，不能提交真实数据、token、图片、
HTML、缓存、checkpoint 地址或 `.env`。本目录 `data/`、`outputs/` 已被 Git 忽略。
`TRAIN_MANIFEST` 使用已经校验的 Train800 标签清洗版，`DEV_MANIFEST` 使用固定 Dev100；
缓存中的 teacher 输入只包含图片和固定 prompt，绝不包含 gold HTML。

```bash
KD_PACKAGE=modeling.llm_post_training.vlm_table_extraction_lab

# 1. 无付费调用的预检；HF 可能下载公开 tokenizer/processor 小文件。
uv run --no-sync python -m "$KD_PACKAGE.kd_collect" \
  --train-manifest "$TRAIN_MANIFEST" --dev-manifest "$DEV_MANIFEST" \
  --data-root "$DATA_ROOT" --cache-dir "$KD_CACHE" \
  --tinker-cookbook-dir "$COOKBOOK_DIR" --limit 80 --max-new-examples 8

# 2. 预算获准后：同一条命令追加 --execute --env-file "$ENV_FILE"
#    和 --max-estimated-usd "$COLLECTION_BUDGET_USD"。
#    这是整个 cache 的累计预算，恢复时不是重新获得一份预算。

# 3. 8条/2步烟测；默认不联系 W&B。预检与执行使用不同新 output-dir。
uv run --no-sync python -m "$KD_PACKAGE.kd" \
  --cache-dir "$KD_CACHE" --train-manifest "$TRAIN_MANIFEST" \
  --dev-manifest "$DEV_MANIFEST" --data-root "$DATA_ROOT" \
  --output-dir "$SMOKE_OUTPUT" --tinker-cookbook-dir "$COOKBOOK_DIR" \
  --official-repo "$RD_REPO" --train-examples 8 --batch-size 4 \
  --max-estimated-usd "$SMOKE_BUDGET_USD"
# 验证 estimate 后，用新的目录追加 --execute --env-file "$ENV_FILE"。
```

烟测确认真实 API 返回 `[N,K]` logprobs、KL 有限且下降、完整 Dev NLL 可重算后，
先通过现有 `evaluate` 跑 teacher 完整 Dev100：指定 `--model Qwen/Qwen3.6-35B-A3B`、
`--revision 995ad96eacd98c81ed38be0c5b274b04031597b0`、相同 pixel/token 上限和 RD scorer。
该命令是实际 inference，**没有默认 preflight 或预算闸门**，需在调用前预留完整费用。
用途是判断 teacher 在本任务上的能力与错误，而非把 teacher 分数当作不可超越的上界。

随后对同一cache运行 collection，使用 `--max-new-examples 80 --skip-rejected`，
复用已完成的7条，只为尚未完成的候选发出请求。pilot训练设置
`--train-examples 80 --skip-rejected --generate-dev --wandb-mode online --dataset-label rd-kd-80candidates-v1`，
仍使用新 output-dir、新建4B LoRA。每个阶段都先根据已有 token 统计重估下一阶段费用。

`--train-examples 80`在过滤模式下表示候选前缀大小；若其中有3条被拒绝，实际训练
数量就是77。结果必须按实际数量解释，不能把候选数当成有效训练数。此次Teacher
Dev100、student训练和最终Test均已完成；过滤策略在训练前固定，没有根据Test挑样本。

最终 Test 使用现有 `checkpoint_eval --stage after --split test`，以 KD 的
`run.json` 为 `--source-run`；不用再支付一次 Base inference。
`log_checkpoint_eval --stage after --baseline-run-id ... --upload` 会重算本地结果，
将模型标记为 `kd`，并放入与已有 Base/SFT800 相同的固定 Test comparison group。

## 日志和费用

W&B 复用隔离 telemetry 子进程，只上传 allowlist 的配置和汇总数值：
teacher/student 型号与 processor SHA、缓存指纹、TopK、温度、LoRA、LR、warmup、
batch、epoch、数据清单哈希、是否生成 Dev、每步 soft CE/entropy/KL、Dev NLL/PPL 与质量指标。
样本行、路径、预测文本和 sampler/state 地址仅写本地；烟测 `wandb-mode=disabled`。

在方案原有预算上，实际实现额外加入首尾两次完整 Train soft-target forward：
增加 `2 × cached_sequence_tokens × $0.33/M`。例如80条、每条含图像prompt后约1600 tokens，
增加约 **$0.0845**；还需另计8条工程烟测。具体训练预检会使用缓存里的真实长度计算。
Teacher rollout 和 Top10 再评分分别计费；Top10 比 Top20 省缓存空间，不代表推理费用减半。

独立阶段的 CLI 上限不会自动合成为本轮总上限；执行者必须累计 teacher collection、
teacher Dev、smoke、pilot 和最终 Test 的实际估计，并预留存储与失败请求费用。
800条扩量及 matched hard-target control 不包含在本次80条 pilot 中。

## MoE 的影响

本实验是**冻结 MoE teacher → dense student**。Teacher 的 router/expert 组合在其内部完成，
student 学习最终 token 概率，不需要对齐专家或复制 router；Tinker 官方已有这种蒸馏示例。

若未来 student 也改成 MoE，输出层 KD 目标仍可相同，但需核实哪些 expert LoRA
会被训练、router 是否可训练、负载均衡策略和 expert 使用情况。`train_mlp=True`
覆盖 MoE MLP，不等于开放 router 训练；当前公开 Tinker API 未发现独立的 router
开关或负载均衡 auxiliary-loss 控制，因此不能宣称这些项已配置。
冻结 router 也不保证路径不变，上游 hidden states 更新后仍可能路由到不同专家。

35B是总语言模型参数，约3B是每token激活参数；并不等于只需要3B模型的显存，
也不能据此保证 dense3B 的延迟。当前Tinker托管teacher免去了我们部署expert的工作。

来源：[官方 MoE→dense TopK 示例](https://tinker-docs.thinkingmachines.ai/tinker/losses/cross-entropy/)、
[Qwen3.6模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)、
[Tinker MoE LoRA说明](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/hyperparam_utils/get_lora_param_count/)。
