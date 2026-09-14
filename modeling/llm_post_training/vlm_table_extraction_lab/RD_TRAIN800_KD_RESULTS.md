# 全量 RD Train800 LoRA KD

2026-09-14。冻结 Qwen3.6-35B-A3B teacher，使用其原始生成轨迹和Top10概率，
从原始hosted Qwen3.5-4B新建rank8 LoRA。全部800张训练图片参与，无样本排除。
训练、完整Dev100与固定Test100均已完成。Test cell/numeric F1均值高于Base，
但仍低于SFT800；整表完全一致率由Base的7%降至3%，不能将这次结果称为全面改善。

- [KD800训练配置与曲线](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/tvsrdjee)
- [KD800固定Test100](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/63278c924c12806f)
- [固定Test100比较组](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/groups/rd-test100-0ddf5237b84f)
- [全量计划和复现入口](FULL_KD_PLAN.md)
- [SFT800结果](RD_TRAIN800_SFT_RESULTS.md)

## 固定 Test100 三版本

Base/SFT800复用已有结果；三者使用相同100张图片、gold标签、prompt、processor/
renderer、greedy解码、8192输出上限和1,048,576 pixels。NLL覆盖108,566个答案tokens。
数字F1平均98条有数字参考的表；其余下表质量指标均覆盖100条，Test raw RD也计分100条。

| Test100 指标 | Base | SFT800 | KD800 |
| --- | ---: | ---: | ---: |
| 单元格 F1 ↑ | 0.4072 | 0.6325 | 0.4538 |
| 数字 F1 ↑ | 0.4445 | 0.7138 | 0.5377 |
| 原版 RD similarity ↑ | 0.8207 | 0.8652 | 0.8373 |
| 格式失败计0的 RD ↑ | 0.7450 | 0.8505 | 0.8169 |
| 格式通过率 ↑ | 0.89 | 0.97 | 0.97 |
| 截断率 ↓ | 0.05 | 0.02 | 0.02 |
| 结构完全一致率 ↑ | 0.14 | 0.37 | 0.19 |
| 整表完全一致率 ↑ | 0.07 | 0.11 | 0.03 |
| NLL ↓ | 0.1431 | 0.0967 | 0.1268 |
| PPL ↓ | 1.1539 | 1.1015 | 1.1352 |

相对Base，cell F1均值增加0.0467、numeric F1增加0.0932；相对SFT800分别低
0.1786和0.1761。这是本次teacher与训练设置下的实测结果，不能推断KD普遍不如SFT。

按图片ID配对，bootstrap 10,000次、seed20260914，得到以下percentile 95% CI。
数字项排除2条无数字参考，持平容差1e-12。

| KD800 − 参考 | 指标 | 平均差值 | 95% CI | 改善 / 持平 / 变差 |
| --- | --- | ---: | --- | --- |
| Base | 单元格 F1 | +0.046688 | [−0.003241, +0.096863] | 54 / 27 / 19 |
| Base | 数字 F1 | +0.093171 | [+0.022648, +0.164446] | 49 / 34 / 15 |
| SFT800 | 单元格 F1 | −0.178626 | [−0.248428, −0.107613] | 25 / 18 / 57 |
| SFT800 | 数字 F1 | −0.176141 | [−0.262290, −0.090221] | 20 / 32 / 46 |

对Base的数字F1提升区间不含0；单元格F1提升区间包含0，证据仍不足。对SFT800的
两项差值区间均低于0。区间仅反映样本层面的不确定性，不涵盖训练/生成重跑、未知
来源聚类或其他文档分布。固定Test反复用于比较，也不等同于全新独立留出集。

## 与 SFT800 对齐的设置

| 项目 | SFT800 | KD800 |
| --- | --- | --- |
| Student | 原始hosted Qwen3.5-4B，新建LoRA | 相同，不继承SFT或KD77 adapter |
| 训练图片 | 固定Train800 | 完全相同800张，顺序相同 |
| LoRA | rank8，attention/MLP开启，unembedding关闭 | 相同 |
| Epoch / batch / steps | 1 / 8 / 100 | 相同 |
| 训练seed | 20260914 | 相同；teacher rollout seed另为20260913 |
| LR / warmup | 1e-4 / 10步，之后恒定 | 相同 |
| Adam | β=(0.9,0.95)，eps1e-8，clip1，weight decay0 | 相同 |
| 监督 | 清洗后的gold HTML，hard CE | 原始teacher轨迹，Top10 soft CE，温度1 |
| 监督答案tokens | 639,203 | 951,723 |
| 计费训练input tokens | 1,118,584 | 1,431,104 |

图片、批次顺序、更新次数和主要超参数已对齐。Teacher答案长度不同，KD监督tokens
约多49%，计费训练input tokens约多28%；这不是等token或等成本实验。
若要单独判断soft概率相对teacher硬答案的作用，需要相同teacher轨迹的hard-target
对照；本轮不包含该额外训练。

Teacher使用固定图片prompt、greedy、关闭thinking，最多8192 output tokens；图像
最多1,048,576 pixels。训练最大序列16384，实际最长完整序列9289。Teacher与student
tokenizer/processor及prompt一致性均有检查；hosted权重revision没有公开，不能完全固定。

## 为什么保留格式失败和截断

此前KD77 pilot选80个候选，按HTML质量过滤丢弃3个，因此不能作为与SFT800的等数据量
对照。用户要求删除其W&B训练/Test记录，已删除并用新查询核验；本地历史结果保留，
不重新发布旧run。

HTML格式通过并不是计算token级KD的必要条件。全量800条中：

- 18条teacher输出未通过严格HTML表格解析，其中包含8条达到8192-token上限的截断；
- 全部800条都具有非空原始tokens和合法Top10概率，均参与训练；
- 不重试或替换图片，不清洗答案，不补EOS，不用gold修补teacher输出；
- 技术性缺失、损坏、词表/概率不合法或超上下文仍会使运行停止。

这让训练图片集合与SFT800一致，也保留了teacher的真实错误；这些错误可能被student
学到。格式合法更不意味着单元格内容必然正确。

## Teacher 缓存与独立核验

复用77份完整teacher缓存与3份旧原始rollout，后者只补Top10；剩余720张各生成一次。
按ID、图片hash、prompt hash重映射索引，旧cache没有修改。采集并发4，共享持久化
reservation账本；所有新增1443次请求均结算，pending为空。

独立核验了全部800张的manifest顺序、图片/prompt身份、原始token decode、原始Top10
响应切片、训练shift与温度1权重，没有增加EOS。旧80条来源整体hash及复用文件hash一致。
Top10保留概率质量：mean0.999846、P05 0.999895、min0.393349；低于0.95的位置占
0.0616%。训练KL针对归一化Top10分布，不是完整词表KL。

采集新增费用按token重算为$2.02583505，其中rollout $1.336000485、Top10再评分
$0.689834565。此前cache累计$0.278309055属于历史费用，不再次计入本轮新增。
Teacher Dev100沿用已有结果，没有重跑。

## 训练监控

每次Dev检查均覆盖相同100条、87,625个gold答案tokens。初始Dev NLL为0.107405，
与SFT800初始0.107404相差约1e-6。初始完整训练soft CE为0.052479，teacher entropy
为0.020457，truncated KL为0.032022；这些与gold Dev NLL使用不同目标分布，不能
直接相减作为train–dev gap，也不把soft CE的指数当作普通perplexity。

前10步实际学习率已核对：第1步1e-5，线性升至第10步1e-4，之后保持1e-4。

| optimizer step | 完整 Dev NLL ↓ | Dev PPL ↓ | 完整训练 soft CE ↓ | 完整训练 truncated KL ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.107405 | 1.113385 | 0.052479 | 0.032022 |
| 25 | 0.083127 | 1.086680 | — | — |
| 50 | 0.083643 | 1.087241 | — | — |
| 75 | 0.081434 | 1.084842 | — | — |
| 100 | 0.085687 | 1.089465 | 0.034531 | 0.014074 |

五次Dev NLL/PPL已从原始逐token log probabilities重算。最后25步gold Dev NLL回升，
虽然最终仍低于训练前；不能宣称训练期间持续改善或没有过拟合。Teacher-target KL
下降与gold NLL回升可能并存，因为两者目标不同，仅凭这些数值不能确定唯一原因。
本轮按预定使用第100步最终checkpoint，不根据Test或中途Dev临时选择其他步数。

最终Dev100生成的cell F1为0.432627、numeric F1为0.504909、格式通过率96%、
截断率3%、结构完全一致率21%、整表完全一致率5%。所有这些指标覆盖100条，
数字项只平均97条有数字的参考表。格式门控RD为0.803121；原版RD为0.836640，
但原版只成功计分99条，不能将它误写成100条的平均值。
缺失的1条为未闭合表格，raw scorer触发`official_alignment_work_limit`；该条在
格式门控RD中仍计0，后者分母保持100。保留原始失败，不通过改写预测补齐分母。

同一Dev上的Base cell/numeric F1为0.382708/0.427688，teacher为0.509603/0.598524，
SFT800为0.633124/0.684234。此次KD改善了部分Base均值，但尚未达到SFT800；
通用teacher在本任务上本就弱于SFT800，模型参数更多不代表这个任务的监督更强。
这是对已测结果的解释线索，不是确定的单一因果结论。

独立审计已核对100个实际更新批次的token数与teacher entropy签名、10步warmup、
五个实现文件指纹及全部Dev结果；W&B的108条日志事件和配置均匹配本地，非空远程
文件仅配置和汇总。Train student逐token forward原始输出未持久化，因此训练CE/KL
没有独立重新执行forward；审计覆盖公式关系、batch对应和teacher entropy。Dev原始
逐token概率与完整预测均可独立重算。

最终三版本Test预测和NLL已独立重算；W&B的配置、全部指标与本地一致，比较组恰好
包含Base、SFT800和KD800。旧KD77训练/Test run均查询不到，非空远程文件只有配置与
汇总。图片、HTML、原始概率、样本行、实际路径及checkpoint地址没有上传W&B或Git。

## 计算费用与耗时

| 本次新增阶段 | Token计算费估算 |
| --- | ---: |
| Teacher rollout + Top10采集 | $2.025835 |
| KD训练、完整训练loss与Dev检查/生成 | $2.411177 |
| 最终Test100 NLL与生成 | $0.233686 |
| **合计** | **$4.670698** |

此前已付费的teacher缓存$0.278309与teacher Dev100 $0.204444在本轮复用，没有
重复计费；若将这两项也计入整套实验，约$5.15。本轮没有重跑Base或SFT800。
金额由实际token数与记录的Tinker单价计算，不是最终账单，不含存储和账单调整。

采集首批约0.5分钟、随后全量采集约21.7分钟；训练含Dev的运行记录窗口约20.3分钟；
最终Test实测305.3秒，约5.1分钟。执行阶段合计约47.5分钟，另有代码准备、预检、
审计与编排时间。训练循环本身711.6秒，包含中途Dev检查，并非纯GPU计算时间。

本地192项lab测试通过（190项首次通过，2项离线W&B服务测试在允许启动本地服务的
环境复跑通过）；Black、Ruff、diff检查及已发布实现的CI通过。
运行的代码提交为`31b7d8e`，内容指纹另覆盖五个KD实现文件；checkpoint/state TTL为7天。
