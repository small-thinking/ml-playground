# 完整 RD Train800 LoRA SFT：首轮结果

日期：2026-09-14。Qwen3.5-4B 使用完整 800 条 RD 训练样本完成 1 epoch 的
LoRA SFT，再在固定 Test100 与原 Base 比较。所有指标由本地逐 token 概率与
完整预测重新计算，W&B 配置、曲线、汇总及文件列表均已读回核对。
Test 单元格与数字 F1 明显提高，结构也改善；仍有 18 张表的单元格 F1 退步。
整表完全一致率由 7% 升到 11%，其差值置信区间包含 0，不能宣称该项提升已确定。

- [训练配置与曲线](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/7oseg76j)
- [固定 Base Test100](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/9409ea84393aebb0)
- [本轮 LoRA Test100](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/8db9fca55b7e19d0)
- [完整配置、预算与复现命令](FULL_SFT_PLAN.md)

## 固定 Test100 对比

复用已经登记的 Base 预测，不再付费重跑 Base。两侧使用相同图片、参考标签、
固定 prompt、processor/renderer、greedy 解码、8,192 输出上限及 1,048,576 pixels。
抽取指标与 NLL 均覆盖全部 100 条；数字 F1 只平均包含数字的 98 条。

| 指标 | Base | Train800 LoRA | 变化 |
| --- | ---: | ---: | ---: |
| 单元格 F1 | 0.4072 | 0.6325 | +0.2253 |
| 数字 F1 | 0.4445 | 0.7138 | +0.2693 |
| 原版 RD similarity | 0.8207 | 0.8652 | +0.0445 |
| 格式失败计 0 的 RD | 0.7450 | 0.8505 | +0.1056 |
| 格式通过率 | 0.8900 | 0.9700 | +0.0800 |
| 输出截断率 | 0.0500 | 0.0200 | -0.0300 |
| 整表完全一致率 | 0.0700 | 0.1100 | +0.0400 |
| 结构完全一致率 | 0.1400 | 0.3700 | +0.2300 |
| NLL ↓ | 0.1431 | 0.0967 | -0.0465 |
| PPL ↓ | 1.1539 | 1.1015 | -0.0524 |

同一张表配对重采样 10,000 次，seed 20260914；下表为平均提升及 percentile 95% CI。
这是样本层面的不确定性，不包含重新训练的随机性，亦不能证明在其他文档分布上成立。

| 指标 | 平均变化 | 95% CI | 改善 / 持平 / 变差 |
| --- | ---: | --- | --- |
| 单元格 F1 | +0.2253 | [+0.1525, +0.2984] | 69 / 13 / 18 |
| 数字 F1 | +0.2693 | [+0.1745, +0.3603] | 58 / 26 / 14 |
| 原版 RD | +0.0445 | [+0.0067, +0.0795] | 57 / 15 / 28 |
| 格式门控 RD | +0.1056 | [+0.0504, +0.1629] | 58 / 16 / 26 |
| 整表完全一致 | +0.0400 | [-0.0200, +0.1000] | 7 / 90 / 3 |

## Train / Dev 监控

每个 Dev 检查点覆盖相同的全部 100 条。完整 Train800 的 NLL 只在首尾计算，
中途不展示旧 Train 值或据此计算 gap。每步 batch NLL/PPL 另记入 W&B；不同
batch 难度不同，不能把单步波动当成完整 Dev 变化。

| optimizer step | 完整 Train NLL | 完整 Dev NLL | Dev PPL |
| ---: | ---: | ---: | ---: |
| 0 | 0.119808 | 0.107404 | 1.113384 |
| 25 | — | 0.055600 | 1.057175 |
| 50 | — | 0.051504 | 1.052853 |
| 75 | — | 0.050711 | 1.052019 |
| 100 | 0.048771 | 0.048820 | 1.050032 |

完整 Dev 自由生成在第 0 / 100 步各执行一次：

| 指标 | 训练前 | 训练后 |
| --- | ---: | ---: |
| 单元格 F1 | 0.3827 | 0.6331 |
| 数字 F1 | 0.4277 | 0.6842 |
| 原版 RD similarity | 0.8162 | 0.8874 |
| 格式失败计 0 的 RD | 0.7615 | 0.8681 |
| 格式通过率 | 0.9200 | 0.9800 |
| 输出截断率 | 0.0400 | 0.0200 |
| 整表完全一致率 | 0.0200 | 0.0900 |
| 结构完全一致率 | 0.1600 | 0.3900 |

NLL/PPL 对 assistant HTML 及结束 token 计算，包含大量可预测的标签语法；PPL 接近
1 不等于抽取内容接近全对。生成侧的单元格、数字、结构与格式指标必须一起观察。
这轮完整 Dev NLL 的变化用于检查是否出现恶化，不能凭一次运行排除所有过拟合。

## 实际执行规格

- 新建 LoRA 初始化，不续训之前的合成 Train8 adapter；rank 8，attention + MLP，
  不训练 unembed；服务接口未暴露 alpha/dropout，记录为空，不猜测默认值。
- 完整 Train800，batch 8，1 epoch，100 optimizer steps；训练输入 1,118,584 tokens。
- Adam：peak LR 1e-4，beta1 0.9，beta2 0.95，eps 1e-8，weight decay 0，grad clip 1。
- 前 10 步线性 warmup：step 1 为 1e-5，step 10 为 1e-4；其后保持 1e-4。
- loss 按 batch 内监督 token 平均；最大完整训练序列 16,384，不截断标签。
- Dev NLL 在 0/25/50/75/100，Dev 生成在 0/100；生成并行度 4，Train 不做自由生成。
- 最终第 100 步 checkpoint 用于 Test，未用 Test 选择学习率、epoch 或 checkpoint。
- 初始/最终 sampler 与最终训练状态 TTL 7 天；本地概率、预测、配置和汇总独立保留。
- 所有图片/标签仅由参数传入；W&B 只记录允许上传的配置、哈希和汇总，无数据文件。

LoRA 使用较高 LR 是经验性起点；本轮未比较多个 LR，不能宣称 1e-4 最优。

## 数据处理与版本

RD 官方资源是评测 benchmark。800/100/100 是本项目的个人实验划分，不能称作
官方训练集或公开 benchmark 得分。此次明确选择 RD Train800；Dev/Test 未更改。
Train 的 798 条标签原样保留，另 2 条仅去掉表格外的文字与外部样式，保留唯一完整
表格、单元格及 rowspan/colspan，图片、顺序和样本数不变。原始及派生标签均不提交。

- 派生 Train manifest SHA-256：`4dcb6476b4495e49fbcd4f0a90fd3e81810ee9bda8128ce339b911e76e6fd003`
- Dev manifest SHA-256：`600e11eae8e92db44e6f16db0bcea0a5ea7251beb8bbb49c035c7d7897e771c1`
- Test manifest SHA-256：`44b9bb3bbfa2b938612f979fec7a6a863761a3c4f67496a1701b5b6b2b2c1ce2`
- 执行 sft.py SHA-256：`ef1e13ae105a450cd5b560ac23ad6884a47f495229f580b304bdcde287facbfd`
- Tinker SDK：`0.27.0`
- processor revision：`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- cookbook revision：`485726f55d3b2b5abe5fcb4a0d2f3e18e4599dfe`
- official scorer revision：`1cae108e6395ddc8389af17385f9769519070558`
- metrics version：`table-eval-v1`

## 时间与费用

| 项目 | 实测时间 | 按实际 tokens 估算的计算费 |
| --- | ---: | ---: |
| 训练 + Train/Dev 评估及保存 | 20.11 分钟 | $2.094636 |
| 其中 optimizer 更新 | 8.42 分钟 | 已包含在上一行 |
| 最终 Test100 NLL + 生成 | 5.36 分钟 | $0.187507 |
| 合计 | 25.47 分钟 | **$2.282143** |

训练时间从本地 preflight 记录建立至 W&B 收尾；Test 时间为评测入口记录的运行时间。
不包含先前代码开发、数据准备及结果分析。计算费按 train $0.737/M、forward
$0.33/M、sample $1.005/M 计算，未计算缓存折扣，**不是供应商账单**。
另预留 $0.10 存储；执行前全流程含 10% 余量预算为 $4.936620，实际计算量低于预算。

## 验证与解释边界

本地自动化检查：223 passed / 3 skipped。独立核验重新计算 7 份完整 NLL 文件
（2 × Train800 + 5 × Dev100）及 2 份 Dev 生成；100 步 LR、输入/监督 token
总量、跨阶段目标 tokens 和所有源文件哈希均一致。最终 Test100 的原始概率、
全部评分、配对差值及费用同样独立重算。

W&B 训练 run 已读回核对全部配置、100 个 batch 点、5 个完整 Dev NLL 点及 2 个
Dev 生成点；最终 Test run 的 16 项指标、训练配置、training run ID 和 Base 分组一致。
两个新 run 的远程文件均只有 config.yaml 与 wandb-summary.json。

单轮结果不是最优参数搜索；hosted 模型权重 revision 无法固定，greedy 也不保证远端
逐 bit 重现。固定 Test100 今后反复查看后应视为回归比较集，不能当作永远未触碰的
最终留出集，也不能外推到没有标签、分布不同的 MLE Interview。后续迭代应依据
Dev 错误提出单一改动，再用相同协议记录结果。
