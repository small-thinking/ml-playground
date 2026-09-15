# Qwen3.6 MoE → Qwen3.5-4B：首次 KD 工程烟测

2026-09-14。真实 Tinker 调用完成，未写入 W&B。这里只验证训练链路，不是80条pilot，
也不宣称表格抽取准确率已经提高。

## 执行与偏离原计划

Teacher为冻结的 `Qwen/Qwen3.6-35B-A3B`，student为新建 `Qwen/Qwen3.5-4B` rank8 LoRA。
Teacher greedy生成，然后在原始token轨迹上再评分获取Top10；student采用τ=1的soft CE。

从预先确定的80条训练子集中尝试前8条：**7条完成缓存，第8条被格式校验拒绝**。
第8条正常结束、没有截断，但生成HTML含不被当前协议接受的 `<img>` 标签。
保留了其原始285个token输出，没有删标签、重新生成或替换样本；未为这条调用Top10再评分。
因此额外明确选择前7条做工程检查，collection仍停在第8条，不能把它称为8条成功。
80条正式训练前必须先解决并记录无效teacher目标的处理策略。

实际训练1 epoch、batch4，共2次optimizer update。LR=1e-4，warmup1步，因此本次
两个更新的LR都是1e-4；计划中的20步pilot将warmup2步，首步5e-5、第二步达到1e-4。
attention/MLP LoRA开启，unembedding关闭；不加载之前的SFT800 adapter。

## 结果

| 指标 | 训练前 | 2步后 |
| --- | ---: | ---: |
| 固定7条 teacher轨迹 soft CE | 0.048177 | 0.038791 |
| 固定 teacher Top10 entropy | 0.021439 | 0.021439 |
| Truncated forward KL | 0.026737 | 0.017352 |
| 完整 Dev100 gold NLL | 0.107389 | 0.094464 |
| 完整 Dev100 gold perplexity | 1.113367 | 1.099070 |

Teacher-target监督位置共8,820个。Top10原始概率质量均值 **0.999865**、P05 **0.999885**；
最低0.830061，低于0.95的比例0.000340。因此总体达到预设覆盖门槛，但不是所有token
都保留接近100%的teacher概率。

Dev前后使用同样100个ID、同样87,625个gold assistant tokens。已从保存在本地的
逐token logprobs独立重算两次NLL/PPL，并核对前后target tokens完全相同。
KL由真实API返回的完整词表student logprobs计算，CE/KL数学另有离线梯度测试。

本次结果表明student在teacher轨迹上的分布拟合改善，同时gold Dev likelihood改善。
没有跑Dev生成、Cell F1、数字F1或Test100，所以还不能据此判断抽取质量或泛化提升。
7条有效样本也不足以估计teacher整体格式失败率。

## 费用与留存

| 阶段 | 计算费用估计 |
| --- | ---: |
| 8条teacher生成 + 7条Top10再评分 | $0.022764 |
| 2步训练 + Train首尾forward + 完整Dev首尾forward | $0.118059 |
| 合计 | **$0.140824** |

这是按记录的token数与单价计算的估计，不是最终账单，未包含checkpoint存储费用。
两个optimizer steps合计约31.02秒；这不包含teacher采集、Dev检查和服务初始化。
已保存最终training state与sampler，TTL为7天；地址仅保存在本地run记录中。

本地输出目录标识为 `rd_kd_moe36_smoke7of8_v1`，含配置、逐步指标、完整Dev概率及核验记录。
运行记录SHA-256：`72b4262a940f9cfe2c24b2d934303822b81e05ea798eabf7ee0b10ead99920b5`。
使用缓存的组合指纹：`212b6c9dc3393fdc2c598e60ad4afdff6b9a47aba6e9fe9ee395f30e4a028854`。
原始图片、HTML、token序列、目标概率、真实路径、凭据与checkpoint地址不提交Git/W&B。

下一步是预算确定后的teacher完整Dev100评测，再决定无效目标策略及80条pilot；
正式student对比仍使用固定Dev/Test协议和已有Base/SFT800结果。
