# 数据分布与 Train/Dev/Test 划分

核验日期：2026-09-13。使用固定种子 `20260913`，在 RD 和 MLE 各自的完整
样本清单上，无放回均匀随机抽取 24 张，人工检查接触表。
逐样本记录见 本地 `manifests/distribution_review.json`（不发布）。

## 抽样与全量尺寸统计

| 观察 | RD-TableBench | MLE Interview |
| --- | --- | --- |
| 抽样量 | 24 | 24 |
| 人工看到的内容 | 24 张以表格/表格式内容为主体，包含无框线、合并、多语言、财务数字、扫描内容 | 8 张清楚含常规数据表；另 1 张手写表单、1 张广告/电视节目单，其他有正文、广告、封面、图表、算法框 |
| 全量竖版图片比例 | 16.0% | 91.5% |
| 宽度中位数 | 1,116 px | 1,191 px |
| 高度中位数 | 432 px | 1,584 px |
| 主要输入差异 | 表格区域 | 整页文档，可能无表格，也可能含多个表格 |

结论：两者都属于文档视觉任务，部分表格内容存在交集，但任务粒度和输入
分布明显不同。MLE 原图不能直接视为与 RD 同分布的表格抽取 Test。
24 张抽样不能精确估计全部类别比例；宽高统计也不是内容分类器。

MLE 没有公开参考标签。即使筛出含表格页面并裁剪，仍需人工参考答案才能
报告有依据的抽取准确率。“面试官持有私有标签”未得到公开证据确认。

因此按用户最新要求划 RD Train/Dev/Test；MLE 作为无标签外部分布检查。
Table Judge 图片/clean HTML 配对可作额外外部检查，但其图片由 HTML 渲染
而来，也不能代表真实扫描件的全部难度。RD 图片通常已定位/裁剪出表格区域，
仍需模型从像素抽取 HTML；对应的 groundtruth HTML 是另外提供的标签。

## 已冻结的 RD 划分

规范文件：本地 `manifests/rd_splits.json`（可用脚本重建，不发布）。

| 集合 | 数量 | 用途 |
| --- | ---: | --- |
| Train | 800 | 完整训练池 |
| Dev | 100 | 所有正式开发评测复用同一组，用于选择模型与实验方案 |
| Test | 100 | 模型和评测协议固定后，比较初始模型与最终模型 |
| train_smoke | 8 | 数据/训练链路 smoke |
| train_10pct | 80 | 第一轮小训练 |
| train_25pct | 200 | 扩大数据量的下一档 |
| train_50pct | 400 | 后续扩展 |

子集是 `8 ⊂ 80 ⊂ 200 ⊂ 400 ⊂ 800`，不是每次重新随机抽样。
Train、Dev、Test 两两没有 ID 重叠。完全相同像素、相同原图 ID、相同单元格文本以及
保守感知相似候选被分组，同组不跨 Train/Dev/Test。

感知候选规则：64 位 dHash 汉明距离不大于 4、宽高比相差不超过 5%。
最终得到 986 个分组，其中 13 个包含多个样本。感知候选尚未逐对人工确认；
宁可保守分组，不把这些候选直接称为已确认重复。原始来源文档 ID 不可得，
这套划分不等于已证明按源文档完全独立，也未完成跨数据集裁剪近重复审计。

当前版本 `rd-train-dev-test-v2` 的 Split SHA-256：
`0c0efa94f0ef82f06980cdc28fb53660bb6c7bbd825373a12982ed6d92c610e0`。
逐条输入清单在 `data/manifests/rd_train*.jsonl`、`rd_dev.jsonl`、`rd_test.jsonl`。
可重复运行 `prepare_splits`；若已有冻结文件与新结果不同，会拒绝覆盖。

这次通过显式 `--revise-from-v1` 迁移，保留了原来的 800 Train 和全部训练
子集；仅将原 Dev 200 按完整分组、种子 `20260914` 分成 Dev 100 / Test 100。
旧划分保存在 本地 `manifests/archive/rd_splits_v1.json`。迁移前
尚无模型推理、训练或基于 Dev 的调参结果，不存在已经看过 Dev 分数再划 Test
的问题。此前对数据的完整性及分布审计，不作为后续模型选择依据。

RD 官方定位是评测集；这里的 Train/Dev/Test 是个人练习拟使用的划分，并不改变
其来源定位或使用条件。尚未执行模型训练。反复用 Dev 选模型后，Dev 结果
应称开发集结果，不能转述为独立泛化测试结果。Test 不用于选择提示词、reward、
超参数或 checkpoint；最终对初始/最终模型各评一次。若再根据 Test 结果调整，
须承认这批 Test 已成为开发数据，不能继续声称它是未见测试集。

## 抽样图

- RD 1–12：`outputs/distribution/rd-tablebench-1.jpg`
- RD 13–24：`outputs/distribution/rd-tablebench-2.jpg`
- MLE 1–12：`outputs/distribution/mle-interview-1.jpg`
- MLE 13–24：`outputs/distribution/mle-interview-2.jpg`

这些衍生图仅用于本地检查，放在被 Git 忽略的 outputs 中。重建命令：

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.inspect_distribution --work-dir "$LAB_WORK_DIR"
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.prepare_splits --work-dir "$LAB_WORK_DIR"
```
