# VLM Table Extraction Lab

低预算的表格 VLM 后训练练习：建立初始模型的抽取 baseline，再观察小规模
SFT、传统 KD、OPD、错误分析和后续迭代带来的变化。目标是学会控制实验与解释指标，不追求
公开榜单最优结果。

当前阶段：Tinker 与本地 MLX 的 4B Dev100 baseline 已完成并保存在
[BASELINE_RESULTS.md](BASELINE_RESULTS.md)。主线默认使用 Tinker；历史推理约
4.7 分钟、token 费估算约 $0.17，MLX 留作可选本地后端。这不是后续运行的固定报价。

第一轮 SFT 使用独立自生成的 8 张训练表格与 4 张验证表格，验证 LoRA 更新、
答案 loss 和生成闭环；配置、运行步骤与结果见 [SFT_EXPERIMENT.md](SFT_EXPERIMENT.md)。
本次 smoke 已跑通 16 步，不调用 W&B；估算计算费用约 $0.0156。简单表格的
cell F1 训练前后均为 1.0，NLL 下降，不代表真实表格能力提升。
smoke 入口默认 peak LR 5e-5 / 10% warmup；每 4 步记录固定 Train/Dev 的 NLL、PPL 和 gap，
每 8 步记录 Dev 自由生成指标。独立 sampler 复核及错配图片对照见实验文档。
warmup 版 SFT 已完成同一轮 before/after 的完整 RD Dev100 对照：NLL
0.1072 → 0.0852、cell F1 0.3765 → 0.4643、数字 F1 0.4177 → 0.5399，
详见 [完整对照结果](RD_DEV100_SFT_RESULTS.md)。这次约 8.9 分钟、计算费估算 $0.40，
仅记录本地结果；目前保留现有划分，之后每个 RD Dev 评估点均覆盖固定全量 100 条。
同一轮模型的完整 Test100 对照也已完成：cell F1 0.4072 → 0.4971、数字 F1
0.4445 → 0.6164、格式通过率 89% → 99%，但整表完全一致率 7% → 3%。
配对统计及完整限制见 [Test100 结果](RD_TEST100_SFT_RESULTS.md)。
首轮完整 RD Train800 LoRA SFT 已完成：rank 8、1 epoch、peak LR 1e-4、10 步
warmup。固定 Test100 的 cell F1 **0.4072 → 0.6325**，数字 F1 **0.4445 → 0.7138**，
结构完全一致率 **14% → 37%**；训练、Dev 和 Test 合计约 25.5 分钟，token 计算费
估算 **$2.28**。完整结果与 W&B 链接见 [Train800 结果](RD_TRAIN800_SFT_RESULTS.md)。

全量传统 Top10 KD 已完成：冻结 Qwen3.6-35B-A3B teacher，4B 从 Base 新建
rank8 LoRA；使用与 SFT800 相同的全部800张图片，训练100步，LR1e-4、warmup10。
固定 Test100 单元格 F1 **0.4538**、数字 F1 **0.5377**，均值高于 Base，仍低于 SFT800；
单元格提升的配对95%区间包含0，数字提升区间不含0。整表完全一致率由7%降至3%。
本轮新增计算费估算 **$4.67**，执行阶段约47.5分钟。三版本可在
[W&B 固定 Test100 比较组](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/groups/rd-test100-0ddf5237b84f)
查看，完整设置、指标、费用和限制见 [KD800 结果](RD_TRAIN800_KD_RESULTS.md)。
旧KD77的W&B训练/Test记录已按用户要求删除，历史结果仅保留作实验记录。

完整 OPD800 已完成，与传统 KD800 使用相同 Base、teacher、800图和100步。
Test100 单元格 F1 **0.4538 → 0.4689**、数字 F1 **0.5377 → 0.5481**、整表一致率
**3% → 4%**，但配对置信区间均包含0；格式通过率 **97% → 95%**，截断率 **2% → 4%**。
因此本轮尚不能确认 OPD 更好。正式训练/Dev/Test 费用估算 **$3.77**，含烟测与失败
请求保守入账约 **$4.04**。评估恢复未重放训练；详情见 [OPD800 结果](RD_TRAIN800_OPD_RESULTS.md)。

RD 是官方评测 benchmark。按本次明确选择，使用个人划分的 Train800 做训练，
Dev100 做开发评估，Test100 做固定比较；这不是官方训练/测试划分。Test100 的首轮留出比较
见 [Test100 结果](RD_TEST100_SFT_RESULTS.md)。按当前约定，每轮迭代都在相同
Test100 上评估，并在 W&B 与 [固定 Base](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/9409ea84393aebb0)
比较；训练过程仍用 Dev。登记命令与分组规则见 [评测流程](EVALUATION.md#固定-test100-baseline-与每轮迭代登记)。
MLE 没有公开任务说明和标签，暂不纳入主线；Table Judge 是独立的裁判校准任务。
这几份资源不是经官方确认的一套训练/测试流程。

推理时 Tinker 接收图片与固定 prompt；SFT 时还会接收训练 HTML 标签以计算 loss。
参考答案不进入生成 prompt。数据、凭证、checkpoint 地址和实际路径仅保留本地。
既有 evaluator 可选上报 W&B 的 quality / structure / runtime 共 14 项汇总指标；
历史 SFT smoke 仅记录本地；正式训练可显式启用 W&B，实时记录 loss/PPL、
学习率、完整 Dev 指标和允许上传的训练配置，不上传原始数据。

- [OPD800 与传统 KD800：结果、区间与费用](RD_TRAIN800_OPD_RESULTS.md)
- [OPD 四个假设：对照设置、预算和新增诊断指标](OPD_HYPOTHESES.md)
- [OPD 固定协议、算法与运行命令](OPD_PLAN.md)
- [OPD 烟测结果与工程修复](OPD_SMOKE_RESULTS.md)
- [传统 Off-policy Top-K KD 方案](OFF_POLICY_KD_PLAN.md)
- [全量 KD800 结果与 Base / SFT800 比较](RD_TRAIN800_KD_RESULTS.md)
- [全量 KD800 配置与复现命令](FULL_KD_PLAN.md)
- [历史 KD77 pilot 结果（W&B记录已删除）](RD_KD80_RESULTS.md)
- [MoE teacher → 4B student：KD 代码与运行步骤](KD_RUNBOOK.md)
- [首次 KD 工程烟测结果：7/8 有效样本、完整 Dev100](KD_SMOKE_RESULTS.md)
- [评测执行、指标与 W&B 隐私](EVALUATION.md)
- [真实 Dev100 baseline 结果](BASELINE_RESULTS.md)
- [完整 Train800 LoRA SFT 结果与 W&B](RD_TRAIN800_SFT_RESULTS.md)
- [完整 Train800 LoRA SFT 配置与预算](FULL_SFT_PLAN.md)
- [第一次 SFT：方案、命令、结果与费用](SFT_EXPERIMENT.md)
- [训练前后完整 RD Dev100 对照](RD_DEV100_SFT_RESULTS.md)
- [训练前后完整 RD Test100 对照](RD_TEST100_SFT_RESULTS.md)
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
| [RD-TableBench](https://huggingface.co/datasets/reducto/rd-tablebench) | 表格抽取 benchmark；1,000 个图片/PDF/groundtruth HTML 配对 | 个人实验划分为 Train800 / Dev100 / Test100；正式首轮使用 Train800 |
| [MLE Interview](https://huggingface.co/datasets/reducto/mle-interview) | 996 张 JPEG，无公开任务说明或标签 | 暂不纳入主线，不能假定其面试任务是表格抽取 |
| [Table Judge Benchmark](https://huggingface.co/datasets/reducto/table-judge-benchmark) | 538 个原图/clean HTML/corrupted HTML 配对及错误元数据 | 独立保留，主要用于 judge 校准；不作为默认训练集 |

RD 评测参考标签来自 `groundtruth/`，不能误用 `providers/`。JPG 与 PDF
表示同一个表格，不是两份独立样本。Judge 的 clean/corrupted 是同一案例，
不能跨 split。保留原始数据，衍生清单不修改源文件。

2026-09-13 已扩大为每个数据集随机 24 张的人工抽查，记录见分布报告。
MLE 不能整体假设为裁剪后的表格测试集；若后续用于表格抽取，需先筛选
含表格页面、定位表格区域，并补充相应任务的参考标签。

RD 数据卡标注 `CC-BY-NC-ND-4.0`；MLE 和 Judge 的当前 HF 元数据没有
明确 license 字段。公开可下载不等于任意用途许可，本目录不再分发数据。
此前 smoke 使用自生成表格；本轮按用户选择使用 RD Train800，数据和派生标签保留本地。

## 验证边界

数据完整性通过不代表标签内容都正确、两套数据语义上不重叠，或模型可运行。
没有标签的 MLE 只能报告格式成功率、长度、耗时及人工抽查结果；模型自己
判断的分数不能冒充真实抽取准确率。

官方 [grading.py](https://github.com/reductoai/rd-tablebench/blob/master/grading.py)
会去掉连字符并允许首尾子表对齐；官方
[convert.py](https://github.com/reductoai/rd-tablebench/blob/master/convert.py)
会展开合并单元格。当前 evaluator 分别报告官方相似度、数字准确性、
完整度、合并结构及输出截断率。
