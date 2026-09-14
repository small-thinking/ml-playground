# Qwen3.5-4B Dev100 baseline：Tinker 与本地 MLX

当前默认推荐 Tinker 4B inference；MLX/Transformers 保留为可选本地后端。
Tinker 的既有 Dev100 推理实测约 4.7 分钟，按 token 与公开费率估算约 $0.17，
并非账单金额。切回默认后端无需重跑已有 baseline，本报告保留两次历史运行。

W&B: [Tinker 运行](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/5eqqv4aa)、
[本地 MLX 运行](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/oo4eqhcq)

冻结的 RD Dev100；未使用 Train 或 Test，也未进行训练。Tinker 结果与本地结果使用相同输入清单和生成上限，推理运行时及 processor 版本不同。Tinker 未暴露权重 revision，无法证明两个 checkpoint 逐位相同。两轮分数差异不是训练收益，也不能严格归因于单一后端因素。

| 指标 | Tinker（既有 baseline） | 本地 MLX |
| --- | ---: | ---: |
| 原版 RD similarity | 0.8167 | 0.8217 |
| RD similarity（格式失败计零） | 0.7615 | 0.7726 |
| 位置敏感单元格 F1 | 0.3827 | 0.4037 |
| 位置敏感数字 F1 | 0.4277 | 0.4484 |
| 格式通过率 | 0.9200 | 0.9300 |
| 截断率 | 0.0400 | 0.0500 |
| 整表完全一致率 | 0.0200 | 0.0300 |
| 结构完全一致率 | 0.1600 | 0.1600 |

- Tinker 推理阶段约 283.2 秒（4.7 分钟）；输入 61,421、输出 151,307 tokens。
  估算采样费 $0.1719，按该次 token 量与公开费率计算；不是账单，也不是以后运行的固定报价。
- Tinker 使用固定 HF processor revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
  与 cookbook revision `485726f55d3b2b5abe5fcb4a0d2f3e18e4599dfe`；服务端权重 revision 未暴露。
- 本地评测耗时：103.8 分钟（包含模型加载及评分，不含首次权重下载和 W&B 上传）。
- 输出 tokens：158,183；输入 tokens：61,421。
- 本地 MLX 推理 API 费用：$0；该次使用本机 Apple Metal，没有 Tinker 或租用 GPU 调用。
- 模型：Qwen/Qwen3.5-4B 原始未量化权重；revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- MLX-VLM 0.7.0、MLX 0.32.2、Transformers 5.17.0；greedy、thinking 关闭、8,192 tokens 上限、1,048,576 pixels 上限。
- 100/100 预测覆盖，100/100 原版 RD 成功计分；数值 F1 的有效样本数见运行 config。
- W&B 已读回验证 finished，仅有 14 项业务指标，分为 quality / structure / runtime；云端文件只有 config.yaml 和 wandb-summary.json。
- 全部原始预测、逐条诊断、清单、凭证和真实路径仅保留本地；未上传到 W&B 或 Git。
  Tinker 推理会发送输入图片和固定 prompt 给服务，参考标签不发送。

## 后续 SFT 对照采用的 Tinker 协议

主线历史基准为 Tinker run `5eqqv4aa`，不是 MLX run。2026-09-14 已核对
本地保存的两份汇总，以上八项指标均与表中四位小数一致；未为核对重新调用模型。

| 固定项 | Tinker 历史 baseline |
| --- | --- |
| 模型与输入 | `Qwen/Qwen3.5-4B`；冻结 RD Dev100；只输入图片与抽取 prompt |
| 生成 | temperature 0、seed `20260913`、thinking 关闭、最多 8,192 output tokens |
| 图片 | 1,048,576 pixels 上限；renderer `qwen3_5_disable_thinking` |
| Processor / cookbook | 使用上文两项固定 revision；不代表托管权重 revision |
| 评分 | evaluator `table-eval-v1`；官方 scorer commit `1cae108e6395ddc8389af17385f9769519070558` |
| 清单 SHA-256 | `600e11eae8e92db44e6f16db0bcea0a5ea7251beb8bbb49c035c7d7897e771c1` |
| Prompt SHA-256 | `fb8dd3e904eb2a09aa8faba5a2a6252cd1b62b72d4176c769d8dbbd1728c5b67`，与当前固定抽取 prompt 一致 |

两轮历史结果的预测覆盖均为 100/100；原版 RD、单元格 F1、格式、截断、
整表与结构指标均在全部 100 条上计算。数字 F1 的有效样本均为 97 条，
其余 3 条两侧都没有数字 token；后续仍按同一规则报告实际有效样本数，
不把 97 硬编码为新预测的分母。失败和截断样本不从主指标分母删除。

合成数据 SFT smoke 首先验证训练、保存 checkpoint 和推理接口能连通；
训练 loss 下降本身不是 RD Dev 质量提升。正式比较时，固定以上评测协议，
记录合成训练数据版本及训练参数，并对同一训练起点的训练前/后 checkpoint
比较全量 Dev。Tinker 历史运行未暴露托管权重 revision，因此该历史基准可作
参考，但不能单凭模型名保证后来训练起点相同；需要保存同次实验的起点证据。
若改评分实现，应使用保存的历史预测按新版本重新评分，再比较分数。
Test100 继续保留到实验方案固定后；W&B 新运行维持 14 项分组指标，完整
分母与逐样本诊断留在本地。此处没有新增训练或评测结果。

## 错误与指标解释

本地格式失败共 7 条：5 条表格未闭合，2 条包含外围文字或不支持的内容。
5 条截断中，4 条与 Tinker 运行重合，另 1 条仅在本地出现；不删除这些失败样本。
单元格指标在 100 条上平均，数字 F1 有效样本为 97 条。

- 原版 RD similarity 采用模糊文本匹配与行列对齐，去掉连字符并宽容首尾缺失；0.82 不代表 82% 单元格正确。
- 格式失败计零是本实验额外定义的严格版本；不符合输出格式时计 0，其余沿用原版评分。
- 单元格 F1 按展开网格的位置和规范化文本精确匹配；同时惩罚遗漏与多出的内容。少一行表头可能使后续位置整体错位。
- 数字 F1 按位置和数字 token 精确匹配，保留负号、括号、百分号及分隔符。它不是数学等价判断。
- F1 是 precision 与 recall 的调和平均。这里先逐表计算再平均，不能直接解释为正确单元格或数字的百分比。

新 W&B run 的 14 项业务指标分为质量 4 项、结构 4 项、运行 6 项；完整诊断留在本地。
原先旧 run 的历史记录保留，未删除。具体映射与执行参数见 [EVALUATION.md](EVALUATION.md)。

历史验证：当时的最终代码重新评分与本轮生成进程的全部汇总一致；当时全量本地
测试为 161 passed、1 skipped，代码 PR CI 通过。此次切回 Tinker 默认值的测试与
合并状态以 PR99 的最新记录为准，不将历史 baseline 冒充新 CLI 的实跑结果。
