# OPD800 与传统 KD800：首轮完整比较

2026-09-14（America/Los_Angeles）。800 张训练图片、100 次 LoRA 更新、完整
Dev100 与固定 Test100 已完成。**OPD 的 F1 均值略高，但本轮没有证据确认它优于
传统 KD；格式通过率反而下降，截断增加。** 本轮主要比较 KD 与 OPD，保留既有
Base/SFT 记录供历史参考。

- [W&B 固定 Test100 比较组](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/groups/rd-test100-0ddf5237b84f)
- [OPD Test100](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/9c1dd8a466614078) · [KD Test100](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/63278c924c12806f)
- [OPD 训练曲线与配置](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/gfab88m4)
- [算法、固定协议、预算与运行命令](OPD_PLAN.md) · [烟测与首次修复](OPD_SMOKE_RESULTS.md)

## 固定条件与算法差别

两者均从 hosted Qwen3.5-4B **Base 新建 LoRA**，teacher 固定为
Qwen3.6-35B-A3B。使用相同 Train800 清单、图片顺序和 shuffle seed20260914；
rank8、1 epoch、batch8、100 updates；Adam LR1e-4、前10步线性 warmup 后恒定，
beta=(0.9,0.95)、eps1e-8、weight decay0、gradient clip1。训练 attention/MLP，
不训练 unembedding；服务未暴露 LoRA alpha/dropout，配置中明确记录为 null。
context16384、最大输出8192、max pixels1048576。

KD 使用固定 teacher 贪心输出及归一化 Top10 分布，最小化 forward-KL 等价的
交叉熵。OPD 每批用当前 student 在 temperature1 下生成一次，teacher 对这些原始
前缀打分；以 `A = log p_teacher - log p_old_student`，通过 native importance
sampling 做一次即时 reverse-KL 更新，按该批 completion token 总数归一化。
无跨批预取、轨迹重放或额外更新，反馈 discount0。

因此这是**两套典型训练方案的比较**：轨迹来源、KL 方向、采样温度及概率估计方式
同时不同，不是只改变 on/off-policy 的单变量消融。训练 KL 数值不可直接横向比较。
主比较检查点在运行前固定为最后第100步；25/50/75步的 sampler 仅保存用于诊断。

## 固定 Test100

两者使用同一份100条 manifest、原图及参考标签、prompt、renderer、官方 scorer，
贪心解码、seed20260913、最大输出8192。数字 F1 的分母是98，其余下表指标为100；
本次两者官方 raw RD 均成功计分100条。

| 指标 | KD800 | OPD800 | OPD − KD |
| --- | ---: | ---: | ---: |
| 单元格 F1 | 0.453844 | 0.468853 | +0.015009 |
| 数字 F1 | 0.537699 | 0.548054 | +0.010355 |
| 官方 RD similarity，原版 | 0.837307 | 0.860057 | +0.022750 |
| RD similarity，格式失败记0 | 0.816918 | 0.826811 | +0.009893 |
| 整表完全一致率 | 3% | 4% | +1个百分点 |
| 结构完全一致率 | 19% | 22% | +3个百分点 |
| 格式通过率 | 97% | 95% | −2个百分点 |
| 输出截断率（越低越好） | 2% | 4% | +2个百分点 |
| Gold assistant NLL（越低越好） | 0.126797 | 0.138925 | +0.012128 |
| Gold assistant PPL（越低越好） | 1.135187 | 1.149038 | +0.013851 |

配对 bootstrap：按固定 manifest 顺序，10,000次重采样，seed20260914；每个指标
重置随机生成器，排除该指标任一侧为 None 的配对，使用百分位95%区间。

| 差值的95%区间 | 下界 | 上界 |
| --- | ---: | ---: |
| 单元格 F1 | −0.034585 | +0.064814 |
| 数字 F1 | −0.055488 | +0.078435 |
| 官方 raw RD | −0.002520 | +0.049225 |
| 格式失败记0 RD | −0.023391 | +0.040756 |
| 整表完全一致率 | −2个百分点 | +5个百分点 |
| 结构完全一致率 | −3个百分点 | +10个百分点 |

单元格 F1：32条改善、36条不变、32条变差。数字 F1：23条改善、47条不变、
28条变差。整表一致：KD 原来正确的3条中丢失1条，新增2条正确，最终为4条；
配对 exact McNemar 双侧 p=1。不能把这些小幅均值变化解释为确定优势。
新增的2条格式失败与新增的2条截断是相同样本；可确认二者在此重合，尚未单独验证因果。

## 完整 Dev 与训练诊断

每个阶段均验证了相同100个唯一 ID、逐条相同的原始参考答案 token，合计87,625个
监督 token。NLL/PPL 由保存的原始 log probability 独立重算。

| 更新步数 | KD Dev NLL | OPD Dev NLL |
| --- | ---: | ---: |
| 0 | 0.107405 | 0.107375 |
| 25 | 0.083127 | 0.101610 |
| 50 | 0.083643 | 0.105572 |
| 75 | 0.081434 | 0.105642 |
| 100 | 0.085687 | 0.100825 |

OPD 在25步后回升，最后又下降；不是持续变好的曲线，也不能仅据此确认过拟合。
第25步相对初始有86/100条 NLL 改善；第50步相对25步有82/100条变差，因此中间
的回升不是单个离群样本造成的。OPD 最终 Dev NLL 高于 KD，但生成 F1 略高，
再次说明 gold likelihood 与自由生成质量是不同的测量。

最终 Dev：KD/OPD 单元格 F1 为0.432627/0.474608，数字 F1 为
0.504909/0.522453；整表一致率5%/4%，结构一致率21%/27%，格式通过率96%/95%，
截断率3%/5%。Dev raw RD 的两侧分母均为99，不能与100条格式门控均值混为一谈。

本轮 OPD 保留全部800条 student 轨迹，包括27条格式不合格、8条截断；没有筛掉困难
输出。共1,002,729个采样 completion token，1,482,110个训练 input token。
每步 learner/sampler importance ratio 均值范围为0.999384–1.000442；均值接近1
不代表每个 token 概率完全一致。审计核验了原始 token、teacher 对齐、shift/mask、
`A/M`、原始 learner 概率及日志；不能独立证明托管服务内部梯度或权重快照身份。

## 费用、时间与评估恢复

| 项目 | token费用估算 |
| --- | ---: |
| 正式 OPD800、5次 Dev NLL、最终 Dev 生成 | $3.545610 |
| 固定 Test100 的 NLL 与生成 | $0.228054 |
| 正式实验合计 | **$3.773664** |
| 烟测，含首次失败请求的保守估算 | $0.265154 |
| 本轮全部合计 | **$4.038818** |

以上按 token 单价估算，**不是供应商账单**，未扣可能的缓存优惠。正式费用已包含
后述8个未观测请求的全额上限$0.067355。原预算未提高：训练/Dev $5.50、Test $1，
烟测 $0.40。KD800 历史新增费用为$4.670698，另复用了更早的 teacher 缓存；成本
账目边界见 [KD800 结果](RD_TRAIN800_KD_RESULTS.md)，不能据此推断普遍价格排序。

实测训练循环57.20分钟（含25/50/75步 Dev），成功的最终 Dev 生成3.72分钟，
Test NLL+生成5.19分钟；这些执行阶段合计约66.10分钟，另有初始化、首尾 Dev NLL、
失败请求、修复与审计时间，不冒充端到端墙钟时间。

全部100步和最后 Dev NLL 已完成后，Dev 生成回执序列化因 SDK
`SampledSequence` 没有 `model_dump()` 而失败。原测试替身错误地模拟了该接口。
已改成读取 SDK 的 tokens/logprobs/stop_reason，并增加真实 SDK dataclass 回归测试。
最终模型已保存，因此仅恢复 Dev 推理，**没有重放任何 optimizer update**。

8个请求没有留下可复核的实际 token 数；保留原失败 run/账本，并按全部8192 token
上限保守计费后，使用原预算补评估。恢复后的100份原始回执与100条预测内容、长度及
停止原因的多重集合一致；回执不含图片 ID，不能独立逐图关联回执。失败与恢复标记
保留在本地及原 W&B 训练历史中。实际训练源码快照与恢复代码哈希分别记录。训练使用提交 `b60dbfc`，
implementation SHA-256 为 `973c81abf676ca19c4c9aa0b689056d37a216e7c3fa5a4cc816e36f281de45f7`；
SDK 序列化修复为提交 `e4dc3dd`。

## 验证、复现与边界

- 本地最终269项测试通过，包括真实 SDK 对象、预算并发/失败行为、数值梯度、图文
  shift/mask、官方评分器、固定 renderer 和 W&B 隐私校验；PR CI 通过。
- 离线审计覆盖全部800图/100更新、5次完整 Dev gold NLL/PPL、最终 Dev 重计分及
  213次账本计数（105次已知调用、8次未知请求保守入账、100次恢复生成），无 pending。
- Test 两侧原始预测重新用同一 scorer 计分，逐条检查数据 hash、manifest、参数及
  108,566个相同 gold target token，再计算上述配对差值。
- W&B 只发布汇总指标及允许的配置。原图、标签、轨迹、概率数组、路径、凭证和
  checkpoint 地址保留本地，未进入 Git 或 W&B。

从 [OPD_PLAN.md](OPD_PLAN.md) 的命令复现训练，实际数据及输出路径通过参数传入。
随后使用 `checkpoint_eval --stage after --split test --expected-examples 100`，
再以 `log_checkpoint_eval --stage after` 发布既有结果。Checkpoint TTL为7天。

本实验只有一个训练 seed，区间只反映这100条样本的变化，未覆盖训练随机性。
反复使用的固定 Test100 也不是每轮全新的独立留出集。Hosted 权重 revision 无法固定。
本轮支持的结论是：**OPD 流程已跑通，部分质量均值略升，但未证明优于传统 KD，
且格式与截断存在退步。** 后续应先在 Dev 上诊断长度/格式问题，再决定是否扩大实验。
