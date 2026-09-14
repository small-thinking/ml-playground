# VLM Table Extraction Lab

低预算的表格 VLM 后训练练习：建立初始模型的抽取 baseline，再观察小规模
SFT、错误分析和后续迭代带来的变化。目标是学会控制实验与解释指标，不追求
公开榜单最优结果。

当前阶段：三份公开数据已下载并完成基础完整性审计；RD 800 Train / 100 Dev / 100 Test
及 8/80/200/400/800 嵌套训练子集已冻结；官方 Table Judge 离线 setup 通过。
训练首轮仍计划选择 Tinker，预留后续 TRL 迁移边界；训练和 GPU 租用尚未启动。
评测已切换为本地推理，支持 Transformers 与 Apple Metal/MLX，使用原始
Qwen3.5-4B 权重。官方评分、补充指标与 W&B 汇总已接入，新 run 仅上报
quality / structure / runtime 三组共 14 项业务指标。

- [评测执行、指标与 W&B 隐私](EVALUATION.md)
- [真实 Dev100 baseline 结果](BASELINE_RESULTS.md)
- [数据完整性记录](DATA_AUDIT.md)
- [分布抽样与数据划分](DISTRIBUTION_AND_SPLITS.md)
- [Table Judge setup](JUDGE_SETUP.md)
- [Tinker 首轮路线、TRL 迁移与 verl 调研](TRAINING_INFRA.md)
- [当前低预算计划](PLAN.md)

## 数据准备

从仓库根目录运行，使用已有 uv 环境。`LAB_WORK_DIR` 和可选 `ENV_FILE`
由调用者设置；数据、manifest 和输出全部保存在指定工作目录中：

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.prepare_data --work-dir "$LAB_WORK_DIR" --env-file "$ENV_FILE" --download
```

省略 `--download` 只重新审计本地数据。脚本固定 Hugging Face revision，
下载时通过显式 `--env-file` 或环境读取可选 `HF_TOKEN`，不自动查找私有路径；不输出 token，
不调用训练、推理或 W&B 服务。本地审计模式不读取 token。

脚本会下载三个数据集的全部仓库文件、解压 ZIP、检查图像解码、配对
RD 图片/PDF/groundtruth、核对 Judge 上游 manifest 中的 SHA-256，并生成
原始文件哈希、样本清单及精确重复检查。下载支持 HF 的续传缓存；ZIP
解压过程校验 CRC。该脚本本身不分配 split；固定划分由 `prepare_splits.py`
生成，规范文件在 `manifests/rd_splits.json`。

```text
data/                         # Git ignored
  raw/                        # 原始 HF 仓库文件，包含 ZIP
  extracted/                  # ZIP 原样解压，包含各 provider 输出
  manifests/
    audit.json                # 核验计数和状态
    raw_files.json            # 原始文件大小与 SHA-256
    rd-tablebench.jsonl        # 图片、人工 HTML 标签、哈希和尺寸
    mle-interview.jsonl        # 图片、哈希和尺寸；无标签
    table-judge-benchmark.jsonl
    exact_duplicates.json
    rd_train*.jsonl            # 带 split/subset 的衍生训练清单
    rd_dev.jsonl               # 固定全量 Dev
    rd_test.jsonl              # 留到最终比较；不用于调参
  tools/table-judge-benchmark/ # 官方源码及其隔离 uv 环境
manifests/rd_splits.json       # Local only: IDs, seed, source hash; no images
manifests/archive/             # 原 800/200 划分的历史版本
outputs/                      # Git ignored; future predictions and checkpoints
```

## 三种数据角色

| 数据 | 官方/可验证内容 | 本实验用途 |
| --- | --- | --- |
| [RD-TableBench](https://huggingface.co/datasets/reducto/rd-tablebench) | 表格抽取 benchmark；1,000 个图片/PDF/groundtruth HTML 配对 | 可划出个人训练/dev/test；一旦用其训练，不声称完整 RD benchmark 的独立测试成绩 |
| [MLE Interview](https://huggingface.co/datasets/reducto/mle-interview) | 996 张 JPEG，无公开标签 | 无标签外部分布检查；无法报告有 ground truth 的抽取准确率 |
| [Table Judge Benchmark](https://huggingface.co/datasets/reducto/table-judge-benchmark) | 538 个原图/clean HTML/corrupted HTML 配对及错误元数据 | 独立保留，可做抽取任务或 judge 任务；分别命名和报告 |

RD 训练标签只来自 `groundtruth/`，不能误用 `providers/`。JPG 与 PDF
表示同一个表格，不是两份独立样本。Judge 的 clean/corrupted 是同一案例，
不能跨 split。保留原始数据，衍生清单不修改源文件。

2026-09-13 已扩大为每个数据集随机 24 张的人工抽查，记录见分布报告。
MLE 不能整体假设为裁剪后的表格测试集；若后续用于表格抽取，需先筛选
含表格页面、定位表格区域，并补充相应任务的参考标签。

RD 数据卡标注 `CC-BY-NC-ND-4.0`；MLE 和 Judge 的当前 HF 元数据没有
明确 license 字段。公开可下载不等于任意用途许可，本目录不再分发数据。
模型训练的具体使用条件在启动训练前核查；必要时以自生成训练数据替代。

## 验证边界

数据完整性通过不代表标签内容都正确、两套数据语义上不重叠，或模型可运行。
没有标签的 MLE 只能报告格式成功率、长度、耗时及人工抽查结果；模型自己
判断的分数不能冒充真实抽取准确率。

官方 [grading.py](https://github.com/reductoai/rd-tablebench/blob/master/grading.py)
会去掉连字符并允许首尾子表对齐；官方
[convert.py](https://github.com/reductoai/rd-tablebench/blob/master/convert.py)
会展开合并单元格。当前 evaluator 分别报告官方相似度、数字准确性、
完整度、合并结构及输出截断率。
