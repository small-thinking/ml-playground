# 低预算实验计划 v4

日期：2026-09-13。状态：数据已下载；RD 800 Train / 100 Dev / 100 Test
与训练子集已冻结；Table Judge 离线 setup 已验证。用户选择先用 Tinker，
以后保留 TRL 迁移路径；付费推理和训练尚未启动。
本计划替代原先约 $90 的完整实验路线，不把旧预算视为已授权。

## 目标与模型选择

只研究同一个初始模型在训练前后的变化，以及哪种数据/训练改动改善哪些错误。
第一轮不做规模 sweep、教师蒸馏或全量 RL。

- 当前首选：Tinker + `Qwen/Qwen3.5-4B`，LoRA，关闭 thinking。2026-09-13 的
  [Tinker 公开模型列表](https://tinker-docs.thinkingmachines.ai/tinker/models/)
  未列出 0.8B 或 2B；最小列出的视觉模型是 4B。不能假定任意 HF 模型
  都能上传到 Tinker 训练。先用 20 条开发样本验证模型与图像输入链路。
- 后续 TRL 候选：[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
  或 [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)。它们支持视觉输入，
  但本项目尚未测量 GPU 显存、吞吐或训练兼容性。若更换模型，必须重新建立
  该模型自己的 baseline，不能把差异全部归因于框架或训练。

小模型的 GPU 小时仍需另付费用；节省 Tinker 余额和节省总现金支出是不同
目标。最少工程/现金投入可以采用 Tinker 4B，但大幅减少训练数据与评测调用。
不在同一个 before/after 对照中更换模型、量化方式、输入分辨率或解码配置。

## 数据顺序

1. 下载 RD、MLE、Judge 的完整公开数据，保留原始文件与版本、哈希。
2. 检查图片/标签配对、图像可读性、输出长度、精确和近重复、来源文档。
3. 按最新要求，划 RD Train 800 / Dev 100 / Test 100。已冻结训练
   子集 8 / 80 / 200 / 400 / 800，彼此嵌套；正式开发评测固定使用全量 Dev。
   Test 不用于挑选提示词、reward、checkpoint 或超参数。实验方案固定后，
   对初始模型和最终模型各评一次 Test；不根据 Test 结果继续挑选模型。
   分组规则、边界和哈希见 [DISTRIBUTION_AND_SPLITS.md](DISTRIBUTION_AND_SPLITS.md)。
4. 保留 Judge 全部案例作外部检查，确认与 RD 训练部分没有可识别重叠。
   MLE 保留作无标签外部输入；最新随机 24 张中有 8 张清楚含常规数据表，
   其余还有正文、广告、封面、表单等。与 RD 的表格区域分布明显不同，
   需要先筛选/定位并补标签，才能讨论有依据的表格测试准确率。
   人工 spot-check 标签，不把 provider 输出当 GT。
5. RD 数据集声明有 NC/ND 条款，训练使用条件仍需核查；若不适合所选用途，
   以 200 条自生成表格作为训练集，RD 保持纯评测。训练不以模糊的许可推断为前提。

## 两个 baseline 任务必须分开

### A. 抽取 baseline（本项目主线）

- 输入：只给原始表格图片及固定抽取指令。
- 输出：只生成保留 `rowspan/colspan` 的表格 HTML，关闭 thinking。
- Judge 数据中的 clean HTML 作为离线参考，不放进模型输入；corrupted HTML
  不参与这个抽取任务。
- 先从开发侧跑 20 条验证链路，正式开发评测固定用全部 100 条 Dev。
  最终量化测试使用 RD Test 100。Judge 图片/clean HTML 可作为额外外部检查，
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

## 首轮预算建议（尚未授权运行）

先执行 A → 80 条 SFT（最多 2 epochs）→ 同协议全量 Dev 复测 → 看失败案例，暂不做 RL。

Tinker 4B 的参考价格：prefill $0.33/M、sample $1.005/M、train $0.737/M。
假设训练序列合计 4,000 计费 tokens，80 × 2 epochs 的训练费约 $0.47。
100 个 Dev 样本各跑训练前/后一次，假设每条输入 3,000、输出 1,000 tokens，
采样费用合计约 $0.40。等实验固定后，100 个 Test 样本对初始/最终模型各评
一次，按相同长度另约 $0.40。以上基础小计约 $1.27，不含额外 Judge 调用。
加上 smoke、开发监控和余量，建议首轮总额度不超过
$5；计划不保证模型质量提升。图片展开 token 数与截断长度需要实测后重算。

首轮不需要租 GPU。以后迁移 TRL 时，再对目标模型做 GPU 可运行性检查，
测峰值显存和吞吐后制定 RunPod 的规格、时长与费用上限。
Tinker 到 TRL 的实现边界及 verl 行业证据见 [TRAINING_INFRA.md](TRAINING_INFRA.md)。
官方 Judge 的安装、离线检查及 provider 边界见 [JUDGE_SETUP.md](JUDGE_SETUP.md)。

首轮预计 4–8 小时准备/评测开发与分析；远端运行时间由 20 样本测速确定。
新 GPU 训练环境搭建时间另算。小 baseline 和单轮 SFT 不需要先建大规模平台。

## 记录与迭代

记录模型 ID、初始 checkpoint、LoRA 参数、图像处理、数据和 split 哈希、
eval IDs、解码配置、训练 tokens、采样 tokens、费用及逐样本错误。
实际运行时接入现有 W&B；本轮不创建远端 run。

数据、提示词、评测和 reward 保持独立于训练后端；只有后端适配代码依赖
Tinker SDK。保存中立的图片/HTML 清单与逐样本预测，不把 Tinker Datum
作为唯一数据格式。TRL 迁移先复用数据和评测，再验证权重与训练语义兼容性；
目前记录的是实现约束，尚未编写训练适配器。

只有在小规模 SFT 与评测可靠运行后，才决定下一步：补数据、改预处理、
继续 SFT 或小规模 GRPO。一次改变一个主要因素，继续 SFT 也应作为 RL 对照。
