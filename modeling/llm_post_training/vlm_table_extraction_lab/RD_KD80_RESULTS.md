# 首轮传统 Top10 KD：77 条有效训练样本

历史pilot记录。用户随后要求全量800条对照，已删除本轮W&B训练与Test记录并核验
不存在；本地结果和teacher缓存保留用于审计与复用，不重新发布。全量运行设置见
[FULL_KD_PLAN.md](FULL_KD_PLAN.md)。下文数值仅描述此次历史pilot。

日期：2026-09-14。固定 Qwen3.6-35B-A3B MoE teacher，先采集其原始生成轨迹和
Top10 概率，再从原始 Qwen3.5-4B 新建 LoRA 训练。没有加载 SFT800 adapter，
没有 student rollout、gold-label CE 混合或 RL reward。

训练、完整 Dev100 及最终 Test100 已完成。KD 的 Test 单元格与数字 F1 高于 Base，
低于 SFT800；原版 RD similarity 略降，格式失败计零后的 RD 提高。

- [当前固定 Test100 比较组](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/groups/rd-test100-0ddf5237b84f)（旧KD77已移除）
- [Teacher 完整 Dev100](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/m7qk5jpg)
- [实现说明与复现流程](KD_RUNBOOK.md)

## 固定 Test100 三版本比较

复用 Base 与 SFT800 的既有预测；相同的100张图片、参考标签、prompt、processor/
renderer、greedy解码和输出/图像上限。数字 F1 平均98条有数字的参考表。
三组按相同协议指纹进入同一个 W&B group，训练数据量和配置另行保留。

| Test100 指标 | Base | SFT800 | KD77 | KD − Base |
| --- | ---: | ---: | ---: | ---: |
| 单元格 F1 | 0.4072 | 0.6325 | 0.4660 | +0.0588 |
| 数字 F1 | 0.4445 | 0.7138 | 0.5265 | +0.0820 |
| 原版 RD similarity | 0.8207 | 0.8652 | 0.8102 | −0.0105 |
| 格式失败计 0 的 RD | 0.7450 | 0.8505 | 0.7931 | +0.0481 |
| 格式通过率 | 0.89 | 0.97 | 0.97 | +0.08 |
| 输出截断率 ↓ | 0.05 | 0.02 | 0.03 | −0.02 |
| 结构完全一致率 | 0.14 | 0.37 | 0.23 | +0.09 |
| 整表完全一致率 | 0.07 | 0.11 | 0.07 | 0.00 |
| NLL ↓ | 0.1431 | 0.0967 | 0.1225 | −0.0206 |
| PPL ↓ | 1.1539 | 1.1015 | 1.1304 | −0.0235 |

这次结果支持“小规模 KD 后部分关键均值提高”，不支持“所有指标提高”。
尤其是原版 RD 与格式门控 RD 方向不同，不能只选有利的一项；整表完全正确率没有变化。

同一张表按ID配对，bootstrap 10,000次、seed 20260914，得到 KD − Base 的平均
差值及 percentile 95% CI。数字项排除无数字参考的2条；持平判断容差为1e-12。

| 指标 | 平均变化 | 95% CI | 改善 / 持平 / 变差 |
| --- | ---: | --- | --- |
| 单元格 F1 | +0.058797 | [+0.006896, +0.114186] | 46 / 28 / 26 |
| 数字 F1 | +0.082021 | [+0.008783, +0.157511] | 38 / 41 / 19 |
| 格式门控 RD | +0.048146 | [−0.002449, +0.104691] | 40 / 27 / 33 |

前两项的区间在此配对重采样下不含0；格式门控RD仍包含0。单元格F1仍有26张表
退步。区间仅反映这100张表的样本不确定性，不包含重复训练的随机性或其他文档分布。

## 数据与训练配置

| 项目 | 本次实际设置 |
| --- | --- |
| Teacher | Qwen/Qwen3.6-35B-A3B，冻结，关闭 thinking |
| Student | Qwen/Qwen3.5-4B；实验 Base 上新建 LoRA，不继承 SFT800 |
| 候选数据 | 固定 Train800 内按 seed 20260913 选 80 条 |
| 实际训练 | 77 条；3 条无效/截断 teacher 输出显式排除，不清洗、重试或补换 |
| Loss | Teacher 原始 token 轨迹上的 Top10 soft CE，温度 1 |
| LoRA | rank 8；attention/MLP 开启，unembedding 关闭 |
| Optimizer | Adam，LR 1e-4，β=(0.9, 0.95)，eps 1e-8，clip 1，weight decay 0 |
| 训练长度 | 1 epoch，batch 4，20 steps，2 步线性 warmup |
| 实测 warmup LR | 第 1 步 5e-5，第 2 步及之后 1e-4 |
| Dev 检查 | 固定全部 100 条；第 0、10、20 步 gold NLL/PPL；最后自由生成 |
| 推理 | greedy，最多 8,192 output tokens，最多 1,048,576 pixels |
| 源码提交 | `e3afc49`；运行中另记录实现内容 SHA-256 |

这里的 Base 是服务商提供的 Qwen3.5-4B，没有经过本项目的表格后训练；并不表示
未经任何 instruction tuning 的预训练权重。Tinker 未暴露 hosted 权重 revision；
本地 processor、tokenizer、renderer、数据、缓存与实现版本均有指纹。

80 个候选均有可核对的最终状态。独立检查了文件哈希、原始 token 解码、完整 Top10
响应与训练切片的对应关系；有效训练共有 108,799 个监督位置。保留概率质量的均值
为 0.999893，P05 为 0.999923；最低为 0.622089，40 个位置低于 0.95。
平均覆盖率很高，但这里的 KL 仍是对归一化 Top10 分布的 KL，不是完整词表 KL。

## 训练与完整 Dev100

| optimizer step | gold Dev NLL ↓ | gold Dev PPL ↓ | 完整训练 soft CE ↓ | 完整训练 truncated KL ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.107388 | 1.113366 | 0.047484 | 0.030427 |
| 10 | 0.083617 | 1.087212 | — | — |
| 20 | 0.080633 | 1.083972 | 0.032097 | 0.015040 |

Dev 每次均为相同的 100 条、87,625 个监督 tokens，已从保存的逐 token log probability
独立重算。训练端 teacher entropy 固定为 0.017057；KL = soft CE − teacher entropy。
soft CE 与 gold NLL 的目标分布不同，不能直接作 train–dev gap，也不把 soft CE
的指数称为普通 perplexity。每步 batch 指标对应不同样本，不能代替完整训练集端点。

三个 Dev 点持续下降，没有观察到后半程 NLL 回升；这不等于已经证明没有过拟合，
也不代表生成指标必然提高。

Teacher 的 Dev 单元格 F1 为 0.509603，数字 F1 为 0.598524，格式通过率 98%，
截断率 2%。它优于已有 Base Dev 的单元格 F1 0.382708，但低于 SFT800 的 0.633124。
Teacher 分数用于理解本任务上的实际教学信号，不能当作 student 的不可超越上界。

| 完整 Dev100 指标 | Base（已有） | MoE teacher | 本轮 KD77 | SFT800（已有） |
| --- | ---: | ---: | ---: | ---: |
| 单元格 F1 | 0.3827 | 0.5096 | 0.4738 | 0.6331 |
| 数字 F1 | 0.4277 | 0.5985 | 0.5165 | 0.6842 |
| 格式通过率 | 0.92 | 0.98 | 0.97 | 0.98 |

本轮 KD Dev 的结构完全一致率为 0.23，整表完全一致率为 0.04，截断率为 0.03。
原版 RD similarity 为 0.831173；将格式失败计零后的 RD 为 0.806850。
Dev 数字 F1 平均97条有数字的参考表；其余质量指标覆盖全部100条。

## 比较边界

KD77 与 SFT800 的训练量和监督来源不同。这是第一轮低成本学习实验，不能据此
断言 soft-target KD 优于或劣于 gold SFT。区分方法收益还需要相同图片、相近 token
预算的 hard teacher-target / soft-target 对照；本次没有额外执行这些训练。

所有训练检查使用 Dev；最终 Test100 不用于挑选 teacher 输出或 checkpoint。
Test 是按本项目约定反复使用的固定比较集；后续频繁据此迭代会降低其独立留出意义。
RD 的 800/100/100 是个人练习划分，不是官方训练/测试划分。Teacher 此次只跑 Dev，
不把它与三个 student 的 Test 指标放进同一比较表。

## 记录与验证

W&B 仅记录允许上传的配置、指纹和汇总指标；原始图片、HTML、概率、样本行、
凭证、实际路径及 sampler/state 地址均保留本地。Base 与 SFT800 复用已有固定
Test 结果，不产生重复推理费用。

开启可选 pinned-renderer / official-scorer 集成后，lab 测试 **174 passed**；
Black、Ruff 与 `git diff --check` 通过。正式采集缓存的 77 个有效和 3 个排除结果
通过独立审计；费用日志 157 次请求、没有待结算 reservation。

训练 W&B 的配置和24条本地日志事件已逐项读回核验，非空远程文件仅配置与汇总。
三版本Test原始预测和NLL均已独立重算；远程group恰好包含这三版，配置、指标与本地
payload一致，非空文件仅配置与汇总。来源运行哈希及固定Test协议指纹也已核对。
Dev 的逐 token 概率可独立重算；Train 的 student forward 原始逐 token 输出没有保存，
因此独立审计对训练 CE/KL 仅检查恒等关系、token 数和固定 teacher entropy，
不声称重新执行过训练 forward。真实二维目标/梯度已在工程烟测验证。
本轮 sampler 与 state 保存时设定7天 TTL；其私有地址仅保存在本地运行记录中。

## 计算费用与耗时

| 阶段 | 本次新增 token 计算费估算 |
| --- | ---: |
| Teacher 完整 Dev100 | $0.204444 |
| Teacher rollout + Top10 采集新增部分 | $0.255545 |
| KD 训练、首尾训练 forward、三次 Dev NLL 与最终 Dev 生成 | $0.520249 |
| 最终 KD Test100 NLL 与生成 | $0.223546 |
| **本次新增合计** | **$1.203783** |

整个80候选 teacher cache 累计估算 $0.278309，其中 $0.022764 是此前烟测已付费并复用的
缓存，不重复计入本次新增。若把这部分历史采集也算入，合计为 $1.226547；这仍不包含
此前独立 student smoke 的费用。Base、SFT800 未重跑。

KD 阶段从记录创建到完成落盘约12.6分钟（含模型初始化、训练与 Dev）；最终 Test
实测291.4秒，约4.9分钟。训练循环记录276.8秒，包含中途Dev检查，不能把它称作
纯GPU训练时间。这些时间不包含前面的 teacher Dev、缓存采集、审计和人工编排。

金额按记录的实际 tokens 与 [Tinker 单价](https://tinker-docs.thinkingmachines.ai/tinker/models/)
估算，未主张缓存优惠，不是最终账单；不包含存储或账单调整。
Teacher 输入/输出为 $0.54/$1.335 每百万 tokens；student forward/输出/train
为 $0.33/$1.005/$0.737 每百万 tokens（2026-09-14核对）。
