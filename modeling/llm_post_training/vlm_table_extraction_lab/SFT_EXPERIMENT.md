# 第一次 SFT：先验证训练闭环

日期：2026-09-14。目标是确认图片 → assistant-only loss → LoRA 更新 →
checkpoint → 图片生成 HTML → 评分这一闭环正确。8 张训练数据上的拟合不是
真实表格泛化收益；本次不上传 Weights & Biases，不运行正式训练。

## 执行顺序

1. 核验并保留 [4B Dev100 baseline](BASELINE_RESULTS.md)，不重复付费生成历史结果。
2. 生成 8 张 Train / 4 张 Dev 的简单表格，图片和 HTML 来自同一份网格。
3. 离线验证哈希、split 隔离、标签有效性、图片 token 保留、答案 mask、
   next-token 对齐、完整目标长度，并计算费用上界。
4. 创建新 LoRA，保存初始 sampler；对 Train8 与 Dev4 测 NLL 并生成 HTML。
5. 训练 16 步，保存最终训练状态和 sampler；在相同输入与解码配置上复测。
6. 本地记录指标、错误及费用；PR 只收录代码、测试与汇总 Markdown。
7. 小实验通过后，估算 80 条正式尝试；由用户审阅后再运行。

## 数据与任务边界

本次使用 `synthetic_data.py`，固定种子 `20260914`。Pillow 从网格生成表格图片，
同一网格序列化为 HTML；包含表头、3/4 列、负数、百分号及不同的行数。
不涉及教师模型付费标注。训练和验证之间检查 ID、图片哈希、标签哈希不相交。
这是同模板下的独立内容验证，并非按模板隔离的泛化测试。

RD-TableBench 官方定位为评估和测试，本次不使用其旧 Train800，也不触碰 Test100。
MLE Interview 公开任务不明确，Table Judge 用于裁判校准，二者均不参与本次训练。
正式学习实验应补充真实/合成的复杂表格覆盖，尤其合并单元格、多层表头和扫描噪声。

SFT 必须将图片、固定指令及答案 HTML 发给 Tinker 计算 loss；生成时只给图片和
固定指令。没有把参考答案放入生成 prompt。所有 PNG、HTML、预测、清单、凭证、
真实路径和 Tinker checkpoint 地址保存在调用者的本地目录，不上传 Git 或 W&B。
本 PR 不包含图片；今后确需提交图片时使用 Git LFS。

## v1 历史参数（保留原实跑条件）

| 参数 | 设置 | 原因 |
| --- | --- | --- |
| 初始化 | `Qwen/Qwen3.5-4B` 上的新 LoRA | 与历史 baseline 同模型 ID；不是名字带 `-Base` 的纯预训练模型 |
| renderer | `qwen3_5_disable_thinking` | 与 baseline 相同，不学习额外 reasoning |
| LoRA rank | 8 | 足够做小规模连通性和拟合检查，不是最优值结论 |
| LoRA 作用域 | attention、MLP 开；unembedding 关 | 首先适配主干；API 未提供视觉塔独立训练开关 |
| LoRA alpha / dropout | API 未暴露，不自行填写 | 不能把常见 PEFT 默认值当作 Tinker 实际参数 |
| batch size | 2 | 8 条每 epoch 4 步 |
| epochs / optimizer steps | 4 / 16 | 有足够重复观察 loss 变化，暂不 sweep |
| Adam | LR `1e-4`，β=(0.9,0.95)，ε=`1e-8` | 保守起点；参数显式记录 |
| weight decay / grad clip | 0 / 1.0 | 小实验不加正则变量，限制异常梯度 |
| LR schedule | constant，无 warmup | 16 步验证先保持简单；正式阶段再决定是否需要调整 |
| loss | assistant HTML + 结束 token 的 cross-entropy | 图片、user 指令、assistant 前缀权重为 0 |
| loss reduction | 整个 batch 的监督 token mean | Tinker 底层求和；本地把权重除以 batch 监督 token 总数 |
| 图像上限 | 1,048,576 pixels | 与 baseline 一致，使用同一个预处理函数 |
| 训练序列上限 | 8,192 tokens | 超长直接拒绝，不截断目标 HTML |
| smoke 生成 | greedy、2,048 tokens 上限、固定 seed | 小表格足够；此处不是 Dev100 的 8,192-token 协议 |
| checkpoint | 初始/最终 sampler、最终 training state；TTL 7 天 | 同一初始化的前后对照；不自动续训或选 best checkpoint |
| 预算 | 默认 token 估算上界不得超过 $0.50 | 本次预检查实际更低；不是服务端硬性扣款上限 |

Tinker 的 LoRA 创建接口不暴露 alpha、dropout、dtype 或逐层 target-module
清单；文档只记录可控参数，不声称与 TRL/PEFT 相同。hosted weight revision
仍不可固定；本次用同一训练客户端保存的初始/最终 sampler 控制初始化来源。

## 指标与通过标准

训练日志只记录每步 assistant token NLL、监督/总 token 数、学习率、耗时、
服务端 optimizer 指标。NLL 是 teacher forcing 下的平均负对数似然，越低越好，
不能直接等同于生成准确率。对固定的完整 Train8/Dev4，额外做训练前后的 NLL；
避免把不同 batch 的首步/末步 loss 当作同一批数据的对照。

生成复用现有 evaluator：原版 RD similarity、格式失败计零的 RD similarity、
位置敏感 cell F1 / numeric F1、格式通过率、截断率、结构/整表完全一致率。
完整诊断存本地。这里的 RD similarity 是**在合成数据上使用 RD 评分方法**，
不是 RD-TableBench 成绩，也不是 Table Judge 的判分。

工程通过标准：16 次优化成功，loss 有限，完整 Train8 的 NLL 下降，保存并加载
最终 sampler 后能够生成可评分的 HTML。Dev4 下降/上升都如实记录；4 张数据的
变化不能证明泛化收益。若原模型已经把简单表格抽取正确，NLL 下降但 F1 不变也
是有效的训练链路证据。不同 checkpoint 地址本身不是权重已更新的证明。

## 当前默认与训练过程监控（v2）

用户复核后，当前默认 peak LR 调整为 **5e-5**，`--warmup-ratio 0.1`：
16 步中前 2 步分别为 2.5e-5、5e-5，之后保持 5e-5。warmup 步数向上取整；
`--warmup-ratio 0` 可关闭。它是更保守的诊断配置，不是已证明优于 1e-4 的最优值。
同时改了 LR 与 warmup，不能把两次差异归因于单独一个参数。

1e-5 不是所有 SFT 的通用学习率。Tinker 的 [LoRA Primer](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/)
建议 LoRA 通常比 full fine-tuning 使用约 10 倍 LR；
[官方小型 SFT 示例](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tutorials/102_first_sft.py) 使用 2e-4。
因此 v1 的 1e-4 不属于明显异常；但当前任务需要验证，不应把官方经验当作最优超参。

| 指标组 | 固定定义与频率 | 如何解读 |
| --- | --- | --- |
| 每步训练 | assistant token NLL / PPL、LR、梯度范数、监督与总 tokens、耗时 | 本步更新前的 batch loss；不能直接与另一批数据作泛化对照 |
| 固定 Train/Dev | 更新后每 `--eval-every 4` 步及 step 0/最终步；NLL、PPL、逐样本平均/最大 NLL、零 logprob 比例 | 同样数据的 teacher-forced 轨迹；逐样本指标防止长表掩盖短表 |
| 训练验证差距 | `dev_nll - train_nll`，同一 checkpoint 上比较 | train 降而 dev 持续升是警讯；绝对 gap 也受两组数据难度影响 |
| Dev 自由生成 | 每 `--generate-every 8` 步及前后；cell/numeric F1、RD similarity、结构、格式、截断、输出长度 | 判断模型能否从图片生成正确内容，不给参考答案 |
| Train 自由生成 | 仅初始与最终 | 检查拟合，降低频繁生成开销 |

16 步对应 NLL 检查 0/4/8/12/16，Dev 生成 0/8/16，Train 生成 0/16。
最终步不重复评估，即使总步数不是间隔的整数倍也会评估。生成间隔必须为 NLL
评估间隔的整数倍。Dev 只做 forward/生成，不参与 backward 或 optimizer step。

NLL = `-sum(supervised target logprobs) / supervised token count`，只包括 HTML
答案与结束 token。PPL = `exp(聚合后的 NLL)`，不是逐样本 PPL 的平均；不同 tokenizer、
mask 或数据集的 PPL 不宜直接比较。PPL 接近 1 表示给定真实答案前缀时目标 token
很容易预测，不等于整张图片的抽取准确率接近 100%。不记录无法从 target logprobs
得到的 top-1 token accuracy。非有限/正 logprob、非二值 mask 或长度不匹配会报错；
PPL 溢出写 null 和显式标记，不伪造截断后的 PPL。

本轮只观察完整曲线，不自动早停或声称选出 best checkpoint。正式训练可在稳定且
足够大的 Dev 上预先定义 checkpoint 选择/early stopping；低 NLL 还要满足生成质量
与格式没有退化。4 张同模板 Dev 不足以排除对模板的过拟合；需要新模板/复杂度的
独立验证和固定 RD Dev100 检查。RD Test100 仍不用于调参。

## 运行命令

从仓库根目录，先同步既有依赖：`uv sync --locked --extra tinker --extra table-eval`。
调用者设置以下变量，实际路径不要写进代码或提交：
`SYNTHETIC_DIR`、`PREFLIGHT_DIR`、`SFT_RUN_DIR`、`TINKER_COOKBOOK_DIR`、
`RD_OFFICIAL_REPO`、`ENV_FILE`。后两个源码目录沿用评测已验证的固定版本。

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.synthetic_data \
  --output-dir "$SYNTHETIC_DIR" --train-count 8 --dev-count 4 --seed 20260914

uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.sft \
  --train-manifest "$SYNTHETIC_DIR/train.jsonl" --dev-manifest "$SYNTHETIC_DIR/dev.jsonl" \
  --data-root "$SYNTHETIC_DIR" --output-dir "$PREFLIGHT_DIR" \
  --tinker-cookbook-dir "$TINKER_COOKBOOK_DIR" --official-repo "$RD_OFFICIAL_REPO"

uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.sft \
  --train-manifest "$SYNTHETIC_DIR/train.jsonl" --dev-manifest "$SYNTHETIC_DIR/dev.jsonl" \
  --data-root "$SYNTHETIC_DIR" --output-dir "$SFT_RUN_DIR" \
  --tinker-cookbook-dir "$TINKER_COOKBOOK_DIR" --official-repo "$RD_OFFICIAL_REPO" \
  --env-file "$ENV_FILE" --learning-rate 5e-5 --warmup-ratio 0.1 \
  --eval-every 4 --generate-every 8 --execute
```

不带 `--execute` 时不创建 Tinker 客户端，也不读取 `.env`；首次 renderer 加载
可能从 HF 下载公开 tokenizer/processor。`HF_HOME` 可由调用者指定缓存，缓存齐全
时可设置 `HF_HUB_OFFLINE=1`。需要固定版本的干净 cookbook checkout，复用
[tinker_inference.py](tinker_inference.py) 中的校验，不运行整个 cookbook 训练框架。

输出目录必须为空；发生异常会记录 `status=failed`，本程序不自动重复训练或恢复。
`run.json` 保存参数、哈希、版本、阶段结果和 checkpoint 地址；`steps.jsonl`
保存每步指标；`evaluations.jsonl` 保存按 optimizer step 对齐的 Train/Dev 曲线。
`*_likelihoods.json` 保存每条样本的监督 target tokens/logprobs，可离线重算 NLL；
`before/after/step_*_predictions.jsonl` 与 `*_details.json` 保存生成诊断。
所有这些原始数据仅在本地，W&B 保持关闭。
日志是本地私有工作文件，不能直接作为 PR 附件上传。

## v1 结果、独立复核与后续成本

本地 run `sft_smoke_v1` 已完成，Tinker SDK 0.27.0，16/16 步成功；未创建 W&B run。
新 LoRA 的初始化、processor、cookbook 与配置按上表执行。正式 80 条训练没有启动。

| 指标 | Train8：训练前 → 后 | Dev4：训练前 → 后 |
| --- | --- | --- |
| Assistant token NLL | 0.150465 → 0.000003323 | 0.138997 → 0.000004006 |
| 原版 RD similarity / 格式失败计零 | 1.0 → 1.0 / 1.0 → 1.0 | 1.0 → 1.0 / 1.0 → 1.0 |
| Cell F1 / numeric F1 | 1.0 → 1.0 / 1.0 → 1.0 | 1.0 → 1.0 / 1.0 → 1.0 |
| 格式通过率 / 结构完全一致率 | 100% → 100% / 100% → 100% | 100% → 100% / 100% → 100% |
| 整表完全一致率 / 截断率 | 100% → 100% / 0% → 0% | 100% → 100% / 0% → 0% |
| 输出 token 总数 | 2,294 → 1,276 | 1,030 → 676 |
| 目标 HTML 字符串完全一致（诊断） | 0/8 → 8/8 | 0/4 → 4/4 |

所有生成覆盖率为 100%；numeric F1 的有效样本数分别为 8、4。
字符串完全一致只去掉首尾空白；整表完全一致则比较解析后的单元格与结构。
原模型已能正确读取这些简单表格，SFT 后更贴近目标 HTML 的具体写法。
因此 **工程闭环通过，但没有观察到抽取 F1 提升；不能据此宣称真实表格能力提高**。
完整 NLL 的下降和输出变化是参数更新生效的证据，不能仅凭 checkpoint 路径不同判断。

训练耗时 145.6 秒（16 步）；训练前检查 84.0 秒，训练后检查 52.1 秒。
总墙钟约 289.6 秒（4.8 分钟，按本地最终报告写入时间减开始时间计算，包含客户端
准备及 checkpoint 操作，不含前面的数据生成与离线预检查）。
按实测 token、无缓存折扣估算计算费用 **$0.015585（约 1.56 美分）**，不是账单。
训练部分 $0.006895；其余包括前后 NLL 与生成。未租 GPU。

复现清单哈希：Train `a57d1434adaa494efb8fc9d5288f4e0a1f7b0b1b9a13e10a71d79b2be0234ee5`；
Dev `fbc06ced4641680d354a8bc1fc034a88e64251e0a7b3d1b408316f3139d338d2`。
本地保存了 `run.json`、16 行 `steps.jsonl`、四份生成及诊断文件、`verification.json`；
保留初始/最终 sampler 和最终训练状态，服务端 TTL 7 天。

验证：最终全量本地测试 **174 passed / 3 skipped**；SFT 专项 6/6 passed
（包含真实固定 renderer 的离线集成）。四份保存预测以最终 scorer 重新评分，
全部汇总与运行报告一致；16 步 loss 均有限，训练 token 合计与预检查一致。
Black 与 diff 空白检查通过。PR 的远端 CI 状态单独报告。
实跑期间只补充了 running 状态和参数记录字段，训练/生成/计费逻辑没有改变；
没有为补日志字段重复付费跑一轮。

### 近零 NLL 的独立复核

用户质疑后，通过保存的初始/最终 sampler 的 `compute_logprobs` API 重算完整序列，
使用同一份答案 mask 提取监督位置。这个接口不接收训练 loss 权重，因而可独立检查
训练阶段的归一化是否误缩小了 NLL。原始输入、targets、mask 还与固定 cookbook 的
官方构造函数逐项比较，12 条全部一致；没有监督图片或 prompt，没有只计算 EOS。
Train/Dev 分别监督 1,276/676 tokens，其中结束标记仅 8/4 个。

| 集合 | 训练接口原报告 | 独立 sampler 复核 | 复核 PPL |
| --- | ---: | ---: | ---: |
| 初始 Train8 | 0.1504647 | 0.1495896 | 1.161358 |
| 初始 Dev4 | 0.1389968 | 0.1381990 | 1.148204 |
| 最终 Train8 | 0.000003323 | 0.000003300 | 1.000003300 |
| 最终 Dev4 | 0.000004006 | 0.000004361 | 1.000004361 |

两条服务路径数值不完全相同，差异原因未单独定位；它们均支持“最终 NLL 已近零”的
结论。最终 Train/Dev 的精确零 logprob 比例约 35.34%/35.21%，最大单 token NLL
约 0.000232/0.000387。数值分辨率会影响近零尾数，不能把这些数当作精确的置信度。

负对照：循环错配 4 张 Dev 的图片，保持原标签不变；最终 sampler 的 NLL 从约
0.00000436 升至 **0.60699**，PPL 从约 1 升至 **1.83491**。这表明至少部分预测
依赖图片，与“完全忽略图片、只靠标签前缀”不符；它不单独证明所有潜在泄漏都不存在。
整个复核 28 次只读概率调用，计算费用估算 **$0.00277**，没有更新参数或上传 W&B。
本地证据为 `sft_smoke_v1/nll_independent_audit.json`。

目前最符合证据的解释是：初始模型已经读对简单表格，SFT 学会其固定 HTML 序列化。
这不是测试出高泛化能力；同生成器 Dev 也下降，并不能排除对共同模板的过拟合。

### v2 监控实跑

使用新的小规模运行验证 warmup 与周期评测；不覆盖 v1，也不从 v1 checkpoint 续训。
本地 `sft_smoke_v2_monitored` 已完成，16/16 步、2 步 warmup 正确执行。
每个固定检查点同时计算 Train/Dev，结果如下（NLL 单位 nats/token）：

| Optimizer step | Train NLL | Dev NLL | Dev PPL | Dev cell F1 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.150465 | 0.138997 | 1.149120 | 1.0 |
| 4 | 0.00077213 | 0.00074340 | 1.00074367 | 未生成 |
| 8 | 0.00013221 | 0.00010800 | 1.00010801 | 1.0 |
| 12 | 0.00005671 | 0.00004012 | 1.00004012 | 未生成 |
| 16 | 0.00003143 | 0.00002109 | 1.00002109 | 1.0 |

Train 最终 PPL 为 1.00003143；Dev numeric F1 与格式通过率在 0/8/16 均为 1，
截断率均为 0。每步 batch NLL 可能波动，固定全量 Train/Dev 的 NLL 在这些检查点
持续下降。本轮未出现 train 降而该 Dev 持续升的典型信号，但不能排除模板过拟合。
更小 LR 和 warmup 后仍快速下降，与任务简单的解释一致，不能推导一般任务上的最优 LR。

优化步骤耗时 80.4 秒；含中间评估的训练循环 113.0 秒；
总墙钟约 218.7 秒（3.6 分钟）。计算费用估算 **$0.019938**，非账单；
预检查上界 $0.07159，包含额外监控。加上只读独立复核，本轮新增计算估算约
$0.0227。没有 W&B 上传，也没有正式数据训练。

所有 10 份 Train/Dev NLL/PPL 均从本地监督 logprobs 重新计算匹配，5 份生成结果
重新评分匹配；完整 16 步、warmup LR 与训练 token 合计均核验通过。
最终全量本地测试 **194 passed / 3 skipped**；新增假客户端流程测试确认 Dev 从未
进入 backward，检查点为 0/4/8/12/16、生成安排无重复，费用计算包含全部监控。

### 正式 80 条尝试的预算与设计

v2 保存的 before/after 已完成完整 RD Dev100 的 NLL 与自由生成评估，见
[外部分布评测结果](RD_DEV100_SFT_RESULTS.md)：NLL 0.1072 → 0.0852、cell F1
0.3765 → 0.4643。真实 Dev 并未接近全对；现有划分继续冻结，每个 RD Dev 评估点
均覆盖全量 100 条，不再把合成 Dev4 当成真实分布质量的替代。

本次简单表格已达到指标上限，不建议直接把同模板复制到 80 条。下一轮先准备
80 张更有挑战且允许训练的表格，另留 20 张独立验证；至少覆盖合并单元格、多层
表头、较长数字、布局和渲染变化。先检查 base 有可改善的错误，再锁定训练清单。
RD Dev100 继续用于外部分布开发评估，Test100 不用于调整方案。

初始建议 rank 8 / peak LR 5e-5 / 10% warmup，batch 4、最多 2 epochs（40 步），先不做超参数 sweep。
正式阶段 NLL 每 10 步、Dev 生成每 20 步，避免把频繁监控开销忽略。
不从这个 smoke checkpoint 继续，以相同基础模型创建新 LoRA；保留其初始 sampler，
正式 before/after 使用同一轮保存的两个 sampler。Dev100 保持 baseline 的
1,048,576 pixels、8,192 output tokens、seed 20260913 及同版本评分。
正式样本变长时，先用 tokenizer 预检查是否超限，而不是静默裁剪。

以下是情景估算，不是已经测量的正式训练吞吐或获准的新增运行：

| 部分 | 假设 | 估算 |
| --- | --- | ---: |
| 训练 | 80 × 2 epochs × 1,000–4,000 序列 tokens | $0.12–0.47 |
| 周期 teacher-forced NLL | Train80 + Dev20，0/10/20/30/40 五次，同样长度范围 | $0.17–0.66 |
| Train/Dev 前后及 step20 Dev 生成 | 220 次，平均 input 600 / output 1,500 tokens | $0.38 |
| RD Dev100 同起点前后生成 | 参考既有 Dev100 的实测 token，运行两次 | 约 $0.34 |
| 合计 | 不含额外 judge、重试、存储 | **约 $1.0–1.9** |

建议正式尝试先设 **$2 预算**，根据真实序列及输出上限重新 preflight；若保守上界
超过 $2，应缩减生成检查量或另行调整预算，而不能用平均值冒充硬上限。
若直接把本次短样本扩成 80×2，训练仅约 $0.0345，但缺乏学习价值，不作为推荐方案。

本次每步平均约 9.1 秒；40 步的机械外推约 6 分钟，然而更大的 batch/更长序列、
服务排队会改变时延。正式训练与各项评估暂按 **20–60 分钟**规划；数据准备和人工
检查另算。保存 sampler 的完整 Dev100 评估入口已由 `checkpoint_eval.py` 提供；
正式阶段仍需先补齐复杂数据，再配置训练期间的全量 RD Dev 检查点及其预算，
不能直接把 smoke 的 Dev4 分数与历史 RD Dev100 分数相比。

预检查：Train8 共 2,339 个输入序列 tokens，4 epochs 合计 9,356 tokens；
训练费估算 $0.006895。前后 Train/Dev NLL 共 7,072 forward tokens，前后生成
共 3,192 prefill tokens，按每次最多 2,048 输出 tokens，计算费用上界约 $0.05968。

费率核验日 2026-09-14：train $0.737/M、forward/prefill $0.33/M、sample $1.005/M。
估算对图片展开和被 mask 的输入 tokens 也计费，不把“只对答案计算 loss”误当成
“只为答案付费”。不减去缓存折扣，不包括额外重试、存储或未来价格变化。

## 来源

- [Tinker 模型与价格](https://tinker-docs.thinkingmachines.ai/tinker/models/)
- [Tinker cross-entropy 的求和语义](https://tinker-docs.thinkingmachines.ai/tinker/losses/cross-entropy/)
- [LoRA 创建接口](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/)
- [RD 官方评测用途](https://reducto.ai/blog/rd-tablebench)
