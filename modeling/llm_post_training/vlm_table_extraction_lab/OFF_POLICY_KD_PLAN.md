# 传统 Off-policy Knowledge Distillation：实验方案

2026-09-14。状态：已按用户最新选择将 soft targets 定为 Top10；尚未生成 teacher rollout、训练 KD 或创建 W&B run。
本轮准备独立 PR；不修改仍待审阅的 [Train800 SFT PR #101](https://github.com/small-thinking/ml-playground/pull/101)。
下述金额是新实验估算，不继承上一轮 SFT 的 $5 执行授权。

## 实验问题与边界

希望学习：在固定 teacher 生成的 HTML 轨迹上，让 student 匹配每一步的 teacher
概率分布，是否能改善表格抽取，以及 soft targets 相对只学 teacher 答案有什么收益。
Teacher 固定；rollout 先生成并缓存，训练中不再更新数据分布，不使用 student rollout。
后续的 on-policy distillation（OPD）另做实验。

Tinker 当前不导出完整词表 logits，但支持每个位置的 Top-K token logprobs，以及
`cross_entropy` 的 `(N, K)` targets/weights。官方示例正是 teacher rollout 后重新
teacher-force，再训练 student。因此本方案推荐 **传统 off-policy Top-K soft-target KD**；
它是完整分布蒸馏的近似，不能标作 exact full-vocabulary KL，也不是 hard-label SFT。
若要求完整词表或双方同温度的高温蒸馏，则另立本地 PyTorch/Transformers + GPU 方案。
来源：[Tinker CE/Top-K 示例](https://tinker-docs.thinkingmachines.ai/tinker/losses/cross-entropy/)、
[官方说明：不暴露完整 logits](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/sdft/)。
这里只使用后者对 API 的说明，不采用它的 student rollout、self-distillation 或 teacher 更新。

流程：图片 → teacher 生成 HTML → 冻结原始 token 轨迹 → teacher Top10 概率 → student 学习 → 完整 Dev/Test。

## Teacher 与 student

当前建议优先验证 teacher：`Qwen/Qwen3.6-35B-A3B`，关闭 thinking；
`Qwen/Qwen3.5-9B` 保留为较小 dense 对照。上一版选 9B 时只排除了退休的
Qwen3.5-35B-A3B，遗漏了当前 3.6 的替代型号，此处修正推荐。

35B-A3B 表示总参数约 35B、每 token 激活约 3B。它是 MoE，不能按 35/9 的
参数比例推断 Tinker 价格，也不能把它理解成性能等同于 3B dense。
账户只读 capabilities 确认：3.6-35B-A3B 与 3.5-9B 可见，3.5-35B-A3B 不可见。
两者在 RD 上的真实质量和吞吐尚未测，公开成绩只支持优先验证 MoE，不能保证胜出。

| 官方模型卡指标 | Qwen3.5-9B | Qwen3.6-35B-A3B |
| --- | ---: | ---: |
| MMMU | 78.4 | 81.7 |
| OmniDocBench 1.5 | 87.7 | 89.9 |
| CC-OCR | 79.3 | 81.9 |

这些是官方各自模型卡的报告值，不是我们按固定 non-thinking RD 协议重测的结果。
[9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、
[3.6-35B-A3B 模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)。

| Tinker 每百万 tokens | 9B dense | 35B-A3B MoE | MoE 相对便宜 |
| --- | ---: | ---: | ---: |
| 输入/forward（非缓存） | $0.66 | $0.54 | 18.2% |
| 输出 | $1.995 | $1.335 | 33.1% |

例如同为 61,421 个输入 tokens、100,000 个输出 tokens 的 Teacher Dev100，
9B 约 $0.2400，MoE 约 $0.1667，便宜约 30.6%。输出长度不同会改变实际差额。
[当前模型与价格](https://tinker-docs.thinkingmachines.ai/tinker/models/)。

首轮 student：`Qwen/Qwen3.5-4B` 的 **fresh LoRA**，不从已 SFT800 的 adapter 续训。
这样 Base → gold-label SFT800 与 Base → KD800 是两个可对照的训练分支。
SFT800 → KD 是另一问题（额外数据/额外训练是否改善），以后单独登记。

已完成的无付费兼容性检查：

- 9B teacher tokenizer/processor revision：`c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- Student tokenizer/processor revision：`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- 3.6-35B-A3B tokenizer/processor revision：`995ad96eacd98c81ed38be0c5b274b04031597b0`。
- 三个模型的 `tokenizer.json` 字节 SHA-256 均为
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`。
- vocab/merges、added/special tokens、normalizer、pre/post processor、decoder 一致；
  image preprocessor config 一致。公开 config 的词表大小均为 248,320。
- 账户的只读 server capabilities 已确认 4B、9B、3.6-35B-A3B、397B-A17B 可见；未调用采样或训练。
- 本地 Tinker SDK 0.27.0 已有 `sample(topk_prompt_logprobs=K)` 和 Top-K 返回结构；
  不需要仅为该参数升级既有环境。离线二维 Datum 构造及 toy CE/KL 梯度等价检查通过；
  真实图像 Top-K 及服务端二维 target 反向传播仍需小烟测。
- 既有 `load_renderer` 目前只允许 4B；实现时需显式增加经过验证的 9B、3.6-35B-A3B 与各自 revision，
  不允许任意模型绕过 tokenizer/processor 检查。

[9B config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)、
[4B config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)。
只能固定本地 tokenizer/processor；Tinker hosted 权重 revision 仍无法固定。

## 首轮设置

| 项目 | 设置 |
| --- | --- |
| 数据 | 既有 Train800 / Dev100 / Test100，沿用冻结清单；pilot 从 Train800 固定选 80 条 |
| Teacher rollout | 每图 1 条，greedy，关闭 thinking，最多 8,192 output tokens，图像最多 1,048,576 pixels |
| Teacher 提示词 | 与 student 抽取提示词相同，只含图片与指令；不提供 ground truth |
| 轨迹 | 保留原始生成 token IDs，冻结缓存；teacher/student 看到同一 teacher 前缀 |
| Soft targets | Top10，记录未归一化 logprobs 与原始 probability mass，再在 Top10 内归一化 |
| KD 温度 | teacher/student loss 温度均为 1；rollout temperature=0 是另一参数 |
| Loss | 纯 soft-target CE；首轮不混入 gold hard-label CE，不加入 RL reward |
| Student LoRA | rank 8；attention + MLP，unembed off；alpha/dropout 未暴露，显式记录未知 |
| 优化器 | Adam，LR 1e-4，beta1=.9、beta2=.95、eps=1e-8、weight decay=0、grad clip=1 |
| Pilot80 | 1 epoch、batch4、20 steps、2 步线性 warmup，然后固定 LR |
| 正式800候选 | 1 epoch、batch8、100 steps、10 步线性 warmup，然后固定 LR |
| 归一化 | batch 内有效 assistant positions 的平均；每个位置的 K 个权重之和为 1 |
| 最大完整序列 | 16,384；超长/截断样本不静默截断或伪造结束 token |
| 完整 Dev NLL | pilot 在 0/10/20；800 在 0/25/50/75/100；使用原始 gold 标签 |
| 自由生成评估 | 训练后完整 Dev100、最终完整 Test100；复用已有 Base/SFT800 预测作参考 |
| W&B | 8 条工程烟测默认关闭；获准运行的 pilot/正式实验仅上传允许的配置与汇总 |

`1e-4` 延续上一轮可用起点以减少变量，不认为 SFT 的最佳 LR 自动等于 KD 最佳 LR。
正式参数以 pilot 的梯度、Dev 变化和实际 token 用量校验为前提。

## Loss 的精确定义

固定 teacher 轨迹前缀为 h，teacher 的 Top-K 集合为 A(h)。在温度 1 下：

$$
\tilde q(v\mid h)=\frac{p_T(v\mid h)}{\sum_{u\in A(h)}p_T(u\mid h)},\qquad v\in A(h)
$$

$$
L_{KD}=-\frac1M\sum_{t\in assistant}\sum_{v\in A(h_t)}\tilde q(v\mid h_t)\log p_S(v\mid h_t)
$$

经典 soft-target 蒸馏参见 [Hinton 等，2015](https://arxiv.org/abs/1503.02531)；
只把 teacher 完整答案作为硬标签训练，是另一个 [sequence-level distillation 对照](https://arxiv.org/abs/1606.07947)。

Teacher targets 不求梯度。Student logprobs 来自其完整词表 softmax，不能再仅在
teacher Top-K 内归一化；否则目标函数会变，也无法正确惩罚 student 在集合外的概率。
`CE - H(teacher_topK)` 是截断/重归一化 teacher 分布与完整 student 分布的 KL，
不是原始完整 teacher 分布的 KL。日志分别命名 `kd/soft_ce`、`kd/topk_teacher_entropy`、
`kd/truncated_forward_kl`；不要将 soft CE 的 exp 冒充 gold-reference perplexity。

首轮固定温度 1，因为仅有 selected student logprobs 时不能恢复任意温度下完整
student softmax 的归一化常数。只调高 teacher 温度、student 保持 1，并不等于
传统双方同温度且乘以温度平方的 KD；不把该差异隐藏在参数里。

Top10 原始 mass 的均值、p05、最小值与低于 .95 的比例必须记录。暂以
mean≥.98、p05≥.90 作为 pilot 工程门槛，非普适理论标准；不满足时先检查数据与
概率接口，在接口允许时再考虑 Top20/50，且重新核预算。不能假定 K=10 总是等价于全词表。

## 分步实施与验收

1. **无付费准备（已完成兼容性检查）**：固定版本、清单、cost model，明确区分
   rollout temperature 与 KD temperature。实现缓存读取/校验及纯 loss 单测。
2. **8 条工程烟测（待授权执行）**：从 Train80 取固定 8 条，teacher rollout + Top10
   缓存，student 做 2 个更新；本地记录，验证图像路径、shift、loss mask 与梯度。
   此阶段不用于质量结论；复用缓存，不为恢复日志再次调用 teacher。
3. **Teacher 完整 Dev100（待执行）**：对选定候选使用同一抽取协议计算 cell/numeric F1、结构、
   格式、RD 指标，与 Base4B 和已 SFT4B 比较。Dev 只选 teacher，不给 teacher 标签。
   Teacher 若没有改善价值或生成大量坏表格，暂停扩量；可保留为机制练习，但不能宣称
   它有望提升当前最好的 student。不能直接把 fallback 换成更贵模型继续消费。
4. **Pilot80**：固定 80 个候选样本，一图一轨迹；报告有效率及排除原因。无效 HTML、
   截断、缺少有效 Top-K 的样本不训练，也不暗中补选更简单样本。有效数小于 80 时
   标记真实 N，不宣称完成 Train80；先查原因再决定是否重试同一张图。
   可用 Train 标签审计 teacher 内容质量，但首轮只按预先定义的格式/完整性过滤，
   不按逐样本 oracle 分数挑最好答案，也不把 gold HTML 替换进缓存。
5. **匹配的 hard-target 对照（建议独立预算）**：使用完全相同的 teacher 轨迹、有效
   样本、初始化、steps 和 LR，只把 soft targets 换为原始 teacher token 的 one-hot。
   hard control 的目标必须是实际生成 token，而不是重新取每步 Top1。这样才可归因
   “soft distribution 相对答案 SFT”的收益；与 gold SFT800 比较还混合了目标来源变化。
6. **扩大800**：只有技术与质量门槛通过、按真实 token 重算预算后才执行。最终 Test
   使用预先约定的最终 checkpoint；不利用 Test 调 teacher、K、LR 或筛选数据。

## 实现要求

保持现有评测和清单格式，计划新增三个清晰入口：teacher 生成/审计、Top-K 缓存、
student KD 训练。所有真实数据/输出/checkpoint 路径由参数传入；缓存独立于训练后端，
便于以后迁移到能读完整 logits 的 PyTorch/TRL 实现。缓存仅保留本地 ignored 目录。

重要的正确性检查：

- 原始生成 token IDs、teacher/student 前缀一致；不先清洗 HTML 再 tokenize 后拿旧概率。
- 明确 next-token shift：输入位置 t 预测 t+1；仅对 assistant 及协议规定的结束位置监督。
  Image/prompt positions 可以没有 logprobs，用 dummy token + zero weight，不对它们报错。
- 结束 token 按 renderer 与实际生成结果处理，不能重复添加；截断输出不伪造 EOS。
- Top-K ids 唯一且在词表内，概率有限，padding 权重为 0，实际有效位置不能为空。
- 无 Top-K 时报错停止，**不静默退回 hard-label SFT**。本地 pinned cookbook 的通用
  helper 有该 fallback，且会在 mask 前检查所有位置；不能直接复制用于 VLM。
- 每 position 权重先归一化，再除以 batch 有效 position 数；不按 N×K 再平均一遍。
  现有 `token_mean_batch` / `summarize_nll` 针对 1D hard targets，不能直接用于二维 soft targets。
- 缓存验证模型/版本、tokenizer、图片与 rollout 哈希、mask/shift、K、温度和 prompt hash；
  支持恢复但不能把不兼容缓存当作新配置的产物。
- 失败停止补发并保留已完成记录；预算包含第二次 teacher forward 和各次 Dev 检查。

单测应覆盖：toy 全词表 KL 与 K=V 的梯度一致、K=1 与对应 hard CE 一致、shift/EOS、
图像 None positions、prompt mask、K 与 batch 归一化、无效概率失败、缓存身份及隐私。
远端烟测再验证 native `(N,K)` CE 返回的 loss 与本地 teacher 权重/selected student
logprobs 重算一致。此前 223 项 SFT 测试不能代替这些新增 KD 检查。

## 记录哪些 metrics

| 类别 | 内容 |
| --- | --- |
| Teacher 数据质量 | 生成有效率、截断率、输出 tokens，Train 审计/Dev 的抽取分数 |
| 蒸馏信号 | soft CE、Top-K teacher entropy、truncated forward KL、Top-K mass 分布、有效 positions |
| 模型质量 | 完整 gold Dev NLL/PPL；原有 16 项 Test 汇总及与 Base/SFT800/同轨迹 hard control 对比 |
| 训练配置 | teacher/student model 与本地版本、初始化、轨迹来源=teacher、K、两种温度、LoRA/Adam/LR/warmup、筛选规则、数据/缓存/代码 hash |
| 资源 | 各阶段 teacher input/output、student training/forward/output tokens、耗时、费用估算及估算边界 |

不上传图片、HTML、token IDs、Top-K 数组、逐样本记录、真实路径、凭证或 checkpoint 地址。
W&B 分组应标记实验/数据协议，不能把 teacher 混入 4B student 的同模型提升曲线。

## 费用与耗时估计

下面主表保留原 9B 方案作价格对照；当前推荐的 MoE 情景见本节末尾。Top20 改成
Top10 主要减少返回数组/本地缓存，不把标准 token 费用减半；必须检查 Top10 保留概率质量。

价格在 2026-09-14 核验：9B teacher input/forward $0.66/M、output $1.995/M；
4B student train $0.737/M、forward $0.33/M、output $1.005/M。按标准 token 计费，
不预扣缓存折扣；Top-K 增加返回数据量，不直接把 token 费用乘 K。真实图像 Top-K
计费和吞吐仍需烟测确认。[官方价格](https://tinker-docs.thinkingmachines.ai/tinker/models/)。

现有完整 Train800 prompt 480,181 tokens；Dev 61,421；Test 63,866。
Pilot80 暂按 Train800 的 1/10 估计 prompt 数，选择固定子集后必须精确重算。
下表将 **teacher 每条生成 1k 或 2k tokens 作为假设**，不是实测区间；student Dev/Test
暂借已有 SFT 模型的 93,147 / 108,884 输出 tokens 作代理。Fresh KD 的长度未知。

| 项目 | Pilot80 / 1k | Pilot80 / 2k | Full800 / 1k | Full800 / 2k |
| --- | ---: | ---: | ---: | ---: |
| Teacher Dev100 生成 | $0.240 | $0.440 | $0.240 | $0.440 |
| Teacher Train rollout | $0.191 | $0.351 | $1.913 | $3.509 |
| Teacher 第二次 forward 取 Top10 | $0.085 | $0.137 | $0.847 | $1.375 |
| Student 1 epoch | $0.094 | $0.153 | $0.943 | $1.533 |
| Student 全部计划 Dev NLL + 最终 Test NLL | $0.204 | $0.204 | $0.303 | $0.303 |
| Student 最终 Dev/Test 生成（代理） | $0.244 | $0.244 | $0.244 | $0.244 |
| 计算合计 | **$1.059** | **$1.530** | **$4.490** | **$7.403** |
| 加 10% 余量及 $0.10 存储预留 | **$1.265** | **$1.783** | **$5.039** | **$8.244** |

若用较长的 Base4B 输出长度作 student 代理，上述计算各多约 $0.109（含余量多约
$0.120），因此 pilot 情景可概述为约 $1.3–1.9。单独的 8 条 student 工程烟测、
重试、更换 teacher、hard-target 对照或额外 epoch 未计入主表；需分别预留，不能
把它们算成免费。Teacher 前 8/80 条缓存可在相同配置下复用，不能重复计费重生成。

若 teacher 和 student **所有生成均达到 8,192 上限**，在相同 prompt-token 假设下：

- Pilot80：计算约 $5.889；加余量/存储约 **$6.578**。
- Full800：计算约 $26.885；加余量/存储约 **$29.673**。

因此目前不能承诺新实验低于 $5，更不能拿 1k 输出情景当作硬上限。执行分阶段：
先确定 teacher 验证/烟测预算，获得实际 token 用量后再给 pilot/800 的具体报价和
执行阈值。每阶段发请求前检查余下请求上界与剩余预算；达不到条件就停止，不自动
缩短输出破坏既定评测协议。Teacher 质量不足时，不继续为 800 条生成数据。

计算方法：设 N 为训练图数、P 为其 prompt tokens、O 为全部 teacher rollout tokens。
Teacher Train 两次调用约 `(2P*.66 + O*(1.995+.66) + N*1.995)/1e6`；最后一项是
第二次 forward 的每条一个未使用生成 token。Student 一 epoch 约 `(P+O)*.737/1e6`。
每次完整 Dev gold NLL 用 training forward 约 $0.049152；Test sampler NLL 约
$0.057003（含每条一个未使用 token）。实际渲染后的序列长度用于正式预检。

缓存使用 assistant positions 的 int32 IDs + float32 logprobs：80×1k×10 约
6.4 MB；800×1k×10 约 64 MB，另加索引/metadata。JSON 嵌套数组会大得多。
这些数组及原始 token IDs 不上传 Git/W&B。

时间尚不能精确预测。参考：上一轮 4B 的 100 optimizer steps 用时 8.42 分钟，
完整 Test100 NLL+生成用时 5.36 分钟；teacher 两次前向、Top-K 传输和队列延迟尚未测量。
可为 pilot 的服务执行与核验预留 30–60 分钟观察窗口，这不是性能承诺；800 的耗时
必须在 pilot 后按吞吐重算。代码实现/测试另计，预计约半天量级，取决于图像对齐烟测。

### 推荐 MoE 的对照预算

保持相同输入/输出长度假设和 student 设置，只替换 teacher 单价：

| 含10%余量与$0.10存储 | Teacher均长1k | Teacher均长2k | 全部输出达到8192 |
| --- | ---: | ---: | ---: |
| Pilot80，9B | $1.265 | $1.783 | $6.578 |
| Pilot80，35B-A3B | $1.103 | $1.480 | $5.400 |
| Full800，9B | $5.039 | $8.244 | $29.673 |
| Full800，35B-A3B | $4.145 | $6.590 | $23.320 |

这是每个方案单独验证一个 teacher 的估算；若两个 teacher 都测 Dev，另一份 Dev
调用要另加，不能当成免费。原有排除项（额外烟测更新、重试、hard-target 对照）
同样适用。这些情景不保证运行费用；先测用量再锁定预算。

## 为什么先跑完整 Teacher Dev100

Teacher 分数是固定任务/协议下的参考水平，不是 student 的数学上限。
首要用途是验证它是否真的擅长当前表格抽取，避免花钱蒸馏比现有 student 更差的目标。
还应检查哪些样本改善/变差、格式/截断、输出长度和用量，以确定教什么、是否值得扩量。

对 cell F1 等越高越好的指标，始终在**同一份 Dev100、同一 prompt/图片/解码/评分协议**下计算：

- Student 起点：S0（fresh 4B Base）。
- Teacher 参考值：T。
- KD 后 student：S1。
- Student 提升：S1 − S0；蒸馏后剩余参考差距：T − S1。
- 已有 gold-label SFT800 另列一列，判断 KD 是否超过当前最好的方案，而不只超过 Base。

T−S1 允许为负，不裁剪，也不宣称 student 必须低于 teacher。若记录
`(S1−S0)/(T−S0)`，仅在 T>S0 时有定义，称为“参考差距缩小比例”，不称为知识传输百分比。
Teacher 和 student 平均分相近也可能犯不同错误，必须保留逐样本配对分析。
不能拿 Teacher Dev100 分数减 Student Test100 分数。Dev 用于 teacher/设置选择；
若最终也要比较 Test 上的 teacher gap，必须在 teacher 固定后额外跑 Teacher Test100，
并将这笔费用另计，不用 Test 来选 teacher。

## 本 PR 的交付边界

本 PR 固定研究问题、推荐 teacher、兼容性证据、loss 定义、对照组、预算和验收标准。
尚不包含可运行的 KD 训练入口，不把已有 SFT runner 标记成已支持 soft targets。
用户已选择 Top10 近似；按以上合同实现，付费验证的 teacher 与预算另行确定。
如果必须严格完整词表，则保留实验对照与评测协议，另设计 GPU 后端，而不是悄悄
改成 teacher 答案 SFT 或 OPD。
