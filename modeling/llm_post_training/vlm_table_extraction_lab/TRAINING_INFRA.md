# Tinker 起步与后续 TRL 迁移

核实日期：2026-09-13。训练首轮计划使用 Tinker；以后保留迁移到自租 GPU
与 TRL 的空间。Tinker 与本地 MLX 推理基线均已完成，当前默认评测恢复为
Tinker 4B inference；本地 MLX/Transformers 可选，未启动训练。

## 当前路线

训练候选为 Tinker + Qwen3.5-4B + LoRA + W&B，默认评测也走 Tinker inference，
评分在本地执行。已有 100 条 Dev baseline，下一训练实验可用嵌套训练子集中的
80 条做最多 2 epochs SFT，
复测同一 Dev 并分析错误。Test 100 留到实验方案固定后，对初始与最终模型
各评一次；不根据 Test 选择模型。首轮不直接上 RL。

[Tinker 当前模型列表](https://tinker-docs.thinkingmachines.ai/tinker/models/)
中最小列出的 VLM 是 Qwen3.5-4B，没有列出 0.8B/2B。本实验并不需要最低
4B 才能成立；这是当前托管后端的选择范围。以后在自租 GPU 上可单独实验
更小的 VLM，并为每个模型重新建立 baseline。

已有 Tinker Dev100 推理实测约 4.7 分钟，按 token 与公开费率估算约 $0.17，
不是账单金额。本地 MLX 的同集合历史运行约 103.8 分钟，推理 API 费用为 $0。
当前选择 Tinker 以减少等待时间，不覆盖或删除本地结果，也无需为切换默认值重跑 baseline。

以下为首轮训练与评测的预算假设：
按训练序列 4,000 tokens、80 条 × 2 epochs 和 $0.737/M 估算，训练费约
$0.47；100 Dev 训练前后各一次，假设每条输入 3,000、输出 1,000 tokens，
采样费合计约 $0.40。最后两个固定模型的 Test 同样另约 $0.40。基础小计
约 $1.27，另留 smoke、重试、额外开发评测与 Judge 费用。首轮建议 $5 上限，
尚未授权运行；图片 token 数和生成长度必须先实测，不能把估算当账单。

统一 evaluator 默认 `--backend tinker`，并发 4、输出上限 8,192 tokens，关闭 thinking。
调用者提供固定 commit 的干净 cookbook 源码目录，仅用于 renderer，不需要 pip 安装 cookbook；
SDK 等依赖使用仓库的 `table-eval` 与 `tinker` extras。具体安装与参数见
[EVALUATION.md](EVALUATION.md)。推理服务接收图片和固定 prompt，标签留在本地评分。
W&B 新运行仅接收 14 项分组业务指标及受控配置，不上传图片、HTML、清单或实际路径。

## 留给 TRL 的实现边界

先保证实验材料可复用，再考虑 checkpoint 转换。以下是后续实现约束，
不是已经完成的训练后端；目前没有 Tinker/TRL 训练适配器。

| 边界 | 两个后端共享 | 后端单独处理 |
| --- | --- | --- |
| 数据 | 样本 ID、图片/标签路径及哈希、split、目标 HTML | Tinker Datum 与 TRL dataset/collator 的构造 |
| 输入协议 | 提示词版本、模型/processor revision、图片尺寸策略、chat template、thinking 开关 | 图像编码及 SDK 张量格式 |
| 学习目标 | assistant-only loss mask、样本顺序、有效 batch、目标 token 定义 | forward/backward、梯度累积、优化器调用与损失归一化实现 |
| 推理与评测 | HTML 输出协议、长度/停止规则、固定 eval IDs、独立 evaluator/reward | 采样 API、推理服务和 checkpoint 加载 |
| 实验记录 | 模型来源、LoRA rank/alpha/target modules、配置、种子、数据哈希、预测及 W&B 指标名称 | Tinker token 费用或 GPU 小时、后端版本与 checkpoint URI |

规范数据继续使用现有 JSONL/图片/HTML，不把 Tinker Datum 或远端 checkpoint
作为唯一可恢复材料。预测至少保留 sample ID、原始输出、解析后 HTML、
stop reason、token usage 和 checkpoint 标识，方便同一 evaluator 比较。
仅在后端模块引入 Tinker SDK；数据划分、指标和 reward 不依赖 SDK。

迁移步骤：

1. 在 TRL 上用同一 base model、processor、模板和解码设置重跑固定 Dev
   smoke，核对图片处理、tokenization 和输出协议。先解决输入差异。
2. 从同一个初始模型复做小规模 SFT，核对 mask、loss 归一化、有效 batch
   和 LoRA 目标层。更换后端与更换模型分成两个实验。
3. 若希望从 Tinker 已训练权重继续，单独确认其导出格式是否能由目标
   Transformers/PEFT 版本加载，并在固定样本上比较加载前后输出/分数。
   不预设 optimizer state 可跨框架恢复；必要时仅迁移权重并重建优化器。

因此可以复用数据、评测与实验设计，但不承诺一行配置即可完全等价地切换。
目前无需提前写通用训练平台或同时安装两个训练后端。

## 公司是否一般都用 verl

公开证据支持 verl 有真实且广泛的工程采用，不能据此推断公司使用占比，
更不能断言所有公司的 post-training 都默认用 verl。

[verl 官方项目](https://github.com/verl-project/verl) 由 ByteDance Seed 发起，
列出的采用/贡献方包括 ByteDance、Qwen、Microsoft Research、Amazon、
LinkedIn 等。更具体的公开案例包括 DAPO 和 Seed-Thinking-v1.5 的训练。
“采用和贡献方”不等于这些公司的全部训练都在 verl 上执行。

框架选择取决于工作负载。以下适用场景是基于官方功能的工程判断，不是
市场份额排名：

| 路线 | 官方能力与合适场景 | 本练习的取舍 |
| --- | --- | --- |
| [TRL + Transformers/PEFT](https://huggingface.co/docs/trl/index) | SFT、DPO、GRPO 等；可处理 VLM | 将来在单卡上学习训练脚本、mask、LoRA 与 reward 的优先路线 |
| [verl](https://github.com/verl-project/verl) | 分布式 RL；协调 FSDP/Megatron 训练与 vLLM/SGLang rollout；也有 SFT | 当 rollout 吞吐、多 GPU 调度或 RL 系统成为学习目标时再引入 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 基于 Ray 等组件的可扩展 RLHF/RL 系统 | 另一种自管开源方案，说明行业存在多种路线 |
| [Tinker](https://thinkingmachines.ai/tinker/) | 用户控制训练循环，平台负责分布式计算 | 本轮节省环境与 GPU 运维时间，接受模型列表和接口范围限制 |

GPU 来源、训练框架和平台服务是不同层：公司可以租云 GPU 或用自有集群，
在上面运行开源框架，并封装成内部训练平台；也可以直接用托管 API。
没有可靠证据支持“公司已经基本都转向训练 SaaS”的结论。

补充：verl 官方已发布 [VeRL-Tinker](https://github.com/verl-project/verl-recipe/tree/main/verl_tinker)，
提供 Tinker 兼容接口，把部分现有训练循环接到自己管理的 verl GPU workers。
这说明 Tinker 接口经验也有潜在复用路径；本次没有验证该 recipe 对我们的
Qwen3.5 VLM 图像输入、LoRA 和完整训练链路的兼容性，不把它作为首轮依赖，
也不替代用户选择的后续 TRL 路线。

以后租 GPU 时，先用目标分辨率、输出长度和 batch 做显存与吞吐测试，再
确定机型与时长。SFT 的显存结果不能直接推广到多 rollout GRPO。
