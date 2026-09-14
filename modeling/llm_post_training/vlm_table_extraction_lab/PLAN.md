# 低预算实验计划 v8

日期：2026-09-14。4B baseline 与合成 Train8 smoke 已完成；历史结果见
[BASELINE_RESULTS.md](BASELINE_RESULTS.md) 和 [SFT_EXPERIMENT.md](SFT_EXPERIMENT.md)。
本轮按用户明确选择，已完成 RD Train800 的 LoRA SFT：1 epoch、rank 8、
peak LR 1e-4、10% warmup，并记录 W&B 配置及曲线，再与固定 Base 比较 Test100。
实际训练、Dev 和最终 Test 计算费估算 $2.28；Test cell F1 0.4072 → 0.6325，
数字 F1 0.4445 → 0.7138。详见 [完整结果](RD_TRAIN800_SFT_RESULTS.md)与
[配置及预算](FULL_SFT_PLAN.md)。后续训练需要另定方案，本轮不自动继续。

RD、MLE、Table Judge 属于不同公开资源，不能假设配套。800/100/100 是个人实验
划分，不是官方训练/测试协议；Dev/Test 不进入训练。MLE 暂不进入主线。
本计划替代此前的合成 Train80 设想及更早约 $90 路线。

## 目标与模型选择

只研究同一个初始模型在训练前后的变化，以及哪种数据/训练改动改善哪些错误。
第一轮不做规模 sweep、教师蒸馏或全量 RL。

- 当前首选：Tinker + `Qwen/Qwen3.5-4B`，LoRA，关闭 thinking。2026-09-13 的
  [Tinker 公开模型列表](https://tinker-docs.thinkingmachines.ai/tinker/models/)
  未列出 0.8B 或 2B；最小列出的视觉模型是 4B。不能假定任意 HF 模型
  都能上传到 Tinker 训练。图像推理链路与 Dev100 baseline 已验证。
- 后续 TRL 候选：[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
  或 [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)。它们支持视觉输入，
  但本项目尚未测量 GPU 显存、吞吐或训练兼容性。若更换模型，必须重新建立
  该模型自己的 baseline，不能把差异全部归因于框架或训练。

小模型的 GPU 小时仍需另付费用；节省 Tinker 余额和节省总现金支出是不同
目标。训练可采用 Tinker 4B 并减少训练数据；当前默认评测同样使用 Tinker。
已完成的 Dev100 推理约 4.7 分钟，估算约 $0.17（按 token 与公开费率计算，
非账单）。需要仅用本机计算时，可显式选择 MLX 或 Transformers。
不在同一个 before/after 对照中更换模型、量化方式、输入分辨率或解码配置。

## 数据顺序

1. 下载 RD、MLE、Judge 的完整公开数据，保留原始文件与版本、哈希。
2. 检查图片/标签配对、图像可读性、输出长度、精确和近重复、来源文档。
3. 保留已有 RD Dev100 作为开发集、Test100 作为每轮固定比较基准；两者均不参与 SFT。
   完整 Train800 用于本轮梯度更新，保留既有划分和哈希，不重新分组。
4. 用自生成图片与 HTML 验证 SFT 数据/训练/推理闭环：8 Train、4 Dev，固定种子。
   这只能说明工程链路和简单表格拟合；不能代表真实扫描件、合并单元格的能力。
5. Train800 全部保留；2 条带表格外文字/样式的标签派生为仅含完整表格的版本，
   留存本地转换审计。Judge 保留裁判校准；MLE 在任务与标签明确前不纳入。

## 两个 baseline 任务必须分开

### A. 抽取 baseline（本项目主线）

- 输入：只给原始表格图片及固定抽取指令。
- 输出：只生成保留 `rowspan/colspan` 的表格 HTML，关闭 thinking。
- Judge 数据中的 clean HTML 作为离线参考，不放进模型输入；corrupted HTML
  不参与这个抽取任务。
- 已完成开发侧链路验证与全量 100 条 Dev baseline，后续按同协议复测。
  每轮完成后在固定 RD Test100 上评估并上传 W&B，与已登记 Base 比较。
  Judge 图片/clean HTML 可作为额外外部检查，
  报告其渲染图片分布边界，不能把无标签的 MLE 分数包装成真实抽取准确率。
- 主指标：内容/数字正确性、行列完整度、合并结构、官方 table similarity。
  另报 HTML 有效率、截断率、输出 tokens、耗时和费用。不可只报告 reward。
- 若用 Judge 的 538 个 image/clean HTML 对作为抽取测试，应明确称为
  “Table Judge 数据上的抽取任务”，不是官方 judge benchmark 得分。

### B. Judge baseline（可选支线）

- 输入：图片 + clean 或 corrupted 候选 HTML，判断是否存在指定错误。
- 输出：正确/错误、错误类型；不让案例文件名和 error 元数据泄漏答案。
- clean 与 corrupted 分开提问并随机化顺序，不先给模型看正确答案。
- 统计：corrupted detection recall、clean false-positive rate、balanced
  accuracy、各错误类型的表现，以及解析失败率；无效输出单独计入。
- 从 Judge 自己的开发划分选样调提示词后，剩余案例才能称 held-out judge test。
  作为 A 的外部测试集使用时，不再用它反复改抽取提示词或奖励。
- 该任务测判断能力，不能代替 A 的抽取质量，也不能预设小学生自身是可靠 judge。

## 正式首轮预算

完整 RD Train800 LoRA SFT → 周期完整 Dev100 NLL 与首尾自由生成 → 最终模型
Test100 评测 → W&B 与固定 Base 比较。只运行一轮，不做学习率或 epoch sweep。
训练 + Dev 计算上界 $3.495553，Test 上界 $0.901375；合计加 10% 余量及 $0.10
存储预留为 $4.936620。该预算已满足本轮低于 $5 的执行授权；实测 token 费用
仍是按公开费率估算，不是账单。配置、计费假设及协议见
[FULL_SFT_PLAN.md](FULL_SFT_PLAN.md)。

首轮不需要租 GPU。以后迁移 TRL 时，再对目标模型做 GPU 可运行性检查，
测峰值显存和吞吐后制定 RunPod 的规格、时长与费用上限。
Tinker 到 TRL 的实现边界及 verl 行业证据见 [TRAINING_INFRA.md](TRAINING_INFRA.md)。
官方 Judge 的安装、离线检查及 provider 边界见 [JUDGE_SETUP.md](JUDGE_SETUP.md)。

首轮原计划预计 4–8 小时准备/评测开发与分析；现有远端 Dev100 实测约 4.7 分钟，
该单次观测不保证以后运行耗时。
新 GPU 训练环境搭建时间另算。小 baseline 和单轮 SFT 不需要先建大规模平台。

## 记录与迭代

记录模型 ID、初始 checkpoint、LoRA 参数、图像处理、数据和 split 哈希、
eval IDs、解码配置、训练 tokens、采样 tokens、费用及逐样本错误。
已有 evaluator 支持 W&B 汇总；历史 SFT smoke 仅记录本地 JSON；完整 Train800 训练实时记录 W&B 的
batch NLL/PPL、学习率、周期完整 Dev 指标及训练配置。
独立 evaluator 保持 quality / structure / runtime 共 14 项指标；保存 checkpoint 的
登记使用 16 项（含格式门控 RD 和 NLL/PPL，省略未知 wall time）。当前 Base Test100
已登记，后续每轮新版本使用独立 run 和同一比较 group，不覆盖 Base；具体规则与命令
见 [评测流程](EVALUATION.md#固定-test100-baseline-与每轮迭代登记)。Tinker
推理只接收图片和固定 prompt；SFT 则需要发送训练答案计算监督 loss。

数据、提示词、评测和 reward 保持独立于训练后端；只有后端适配代码依赖
Tinker SDK。保存中立的图片/HTML 清单与逐样本预测，不把 Tinker Datum
作为唯一数据格式。TRL 迁移先复用数据和评测，再验证权重与训练语义兼容性；
当前小规模训练入口为 sft.py，TRL 后端尚未实现。

只有在小规模 SFT 与评测可靠运行后，才决定下一步：补数据、改预处理、
继续 SFT 或小规模 GRPO。一次改变一个主要因素，继续 SFT 也应作为 RL 对照。
