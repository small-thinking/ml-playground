# 小样本 SFT 后的完整 RD Dev100 对照

日期：2026-09-14。比较 `sft_smoke_v2_monitored` 同一训练初始化保存的 before/after
sampler；训练为 8 张自生成表格、16 步 LoRA，peak LR 5e-5、10% warmup。
这次不更新模型参数、不改变数据划分、不访问 Test100，也不上传 W&B。

后续应用户要求，同一轮模型另做了 [Test100 留出对照](RD_TEST100_SFT_RESULTS.md)；
本页保留当时的 Dev 结果和决策，Test 结果不混入本页指标。

## 先区分两套 Dev

- **Synthetic Dev4**：原 SFT smoke 的完整验证集，来自与 Train8 相同的简单生成器。
  每次都评完 4 张，不是从 RD Dev100 中随机挑 4 张。
- **RD Dev100**：原先冻结的真实表格开发集；本报告使用完整 100 条，不筛选成功样本。
- **错配图片负对照**：保持 Dev4 标签不变，故意循环换成其他图片，检查 NLL 是否上升。
  它是另行构造的诊断，既不是正式评测输入错误，也不是某次随机抽样。

扩大样本数会降低估计的不确定性，但不会自动让相同简单分布变难。这里比较的是
两个来源/复杂度不同的集合；不能把结果变化全归因于 4 与 100 的样本数差别。

审计确认：v2 在 0/4/8/12/16 的 NLL 评估都覆盖相同的完整 Dev4（676 个监督 tokens），
0/8/16 的生成也覆盖完整 Dev4。合成 Train8 与 RD Dev100 的 ID、标签字节、图片
字节、解码 RGB 像素交集均为 0。精确重复检查不能证明不存在近重复或基础模型的
预训练污染，但没有证据支持“随机抽了一小部分 Dev”或这些精确重复的解释。

## 协议与可复现入口

沿用冻结的 RD Dev100，manifest SHA-256：
`600e11eae8e92db44e6f16db0bcea0a5ea7251beb8bbb49c035c7d7897e771c1`。
复用相同图片上限 1,048,576 pixels、固定 renderer/processor、greedy、seed 20260913、
最大生成 8,192 tokens，以及同版本官方 RD 和补充评分器。

NLL 在两个保存的 sampler 上通过 `compute_logprobs` 计算；参考 HTML 保持原样，
只统计 assistant 答案及结束 token，完整序列不截断。NLL 的上限检查为模型的
65,536-token 上下文，生成上限仍是 8,192；二者不是同一个限制。
本次最长完整 reference 序列为 9,270 tokens，100 条均完整计入 NLL，没有因长度排除样本。PPL 为全体监督
tokens 的 mean NLL 再取 exp；另保存逐样本概率以便复算。

自由生成只输入图片和固定指令，不给参考 HTML。全部 100 个样本参与评分；
格式错误和截断不删除，数字 F1 按现有规则报告实际有效样本数。

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.checkpoint_eval \
  --source-run "$SFT_RUN_DIR/run.json" \
  --manifest "$LAB_WORK_DIR/data/manifests/rd_dev.jsonl" \
  --data-root "$LAB_WORK_DIR" --output-dir "$CHECKPOINT_EVAL_DIR" \
  --tinker-cookbook-dir "$TINKER_COOKBOOK_DIR" --official-repo "$RD_OFFICIAL_REPO" \
  --env-file "$ENV_FILE" --expected-examples 100 --execute
```

路径通过参数提供；不带 `--execute` 仅做本地预检查。每个阶段最多 4 个并行请求，
逐条落盘；失败后取消未启动的请求，不自动重跑。输出目录必须为空。
计费上界包含 200 次完整 reference NLL 调用及 200 次自由生成，所有生成都用满
8,192 output tokens 时计算费上界约 $1.79；实际费用按实际 token 另算，不是账单。

## 结果

两个阶段的 NLL 和自由生成均完成 **100/100**；每阶段 NLL 包含 **87,625** 个
监督 tokens。前后 target tokens 完全一致；保存的逐 token 概率独立复算 NLL/PPL、
保存的全部预测重新评分，均与报告一致。

| 指标 | Before | After SFT | 变化 |
| --- | ---: | ---: | ---: |
| Assistant NLL（nats/token，低为好） | 0.107236 | 0.085174 | −0.022062 |
| Perplexity（低为好） | 1.113197 | 1.088907 | −0.024290 |
| 官方原版 RD similarity | 0.820289 | 0.835739 | +0.015450 |
| RD similarity，格式失败记 0 | 0.761432 | 0.804817 | +0.043385 |
| Cell F1 | 0.376507 | 0.464304 | +0.087797 |
| Numeric F1 | 0.417704 | 0.539910 | +0.122205 |
| 格式通过率 | 92/100 | 96/100 | +4 个 |
| 表格内容完全一致率 | 2/100 | 3/100 | +1 个 |
| 结构完全一致率 | 16/100 | 19/100 | +3 个 |
| 输出截断率（低为好） | 4/100 | 4/100 | 无变化 |

Numeric F1 两阶段各有 97 个有效样本，其余按既有规则不参与该指标均值；
全部 100 条仍参与整体评测。RD 原版分数与格式门控分数分开报告，不混用口径。
这里的 before 来自本轮保存的初始 sampler，不是把旧的历史 baseline 拼接到新 after。

总耗时 **531.8 秒（8.9 分钟）**；生成输入合计 122,842 tokens，生成输出
260,621 tokens，NLL 完整输入合计 298,092 tokens。按既有费率计入概率 API
额外生成的 200 个 token，计算费估算 **$0.40103**，不代表供应商账单。
没有重新训练，也没有 W&B 上传。

## 解释与下一轮约定

1. **真实 Dev 的 NLL 没有接近零。** 本轮 after 为 0.08517，而合成 Dev4
   after 为 0.00002109。差异不能只用样本数解释；两套数据的复杂度和分布不同。
   NLL 在给定正确前缀的条件下平均所有答案 tokens，容易预测的 HTML 标记等也
   参与平均；PPL 接近 1 不等于整张表格正确。
2. **本次生成质量有可观察的改善，但远未解决任务。** Cell/数字 F1、格式通过率
   同时上升，只有 3 张表格内容完全一致，仍有 4 张截断。仅一个训练运行与一次
   前后生成对照，尚未测量重复运行方差，也未证明换一批数据仍有同等收益。
   本轮不能单凭这些汇总指标把收益归因为更好的视觉识别；序列化、结构对齐和
   输出格式也可能贡献改善，需要后续错误分析区分。
3. **先保留现有划分。** 目前没有因为“Dev 太容易”而重分 RD 的证据；为了分数
   好看或变难而重抽 Dev 会破坏已有比较。若后续发现近重复、共同来源跨集合或
   明确的部署分布差异，再按来源/模板分组重划，并版本化和重跑 baseline。
4. **以后每个 RD Dev 评估点都跑固定全量 100 条。** 小规模训练只缩小 Train，
   不缩小用于模型比较的 RD Dev；训练时明确配置 NLL 与自由生成检查点，每次
   对对应 Dev 全量评分。Synthetic Dev4 只保留作流程诊断，并始终明确标注来源，
   不能代替 RD Dev100 的模型质量结论。Test100 继续留到最终比较。

本地原始证据目录为 `outputs/rd_dev100_sft_v2/`，包含 `run.json`、两阶段
`*_likelihoods.jsonl`、`*_predictions.jsonl`、`*_details.json` 和复核
`verification.json`；只提交本报告的汇总，不提交样本、HTML、token IDs 或 sampler 地址。
本轮仓库测试 **199 passed / 3 skipped**，覆盖概率对齐、完整样本数、费用及失败后
取消排队请求。已有运行使用的并发入口在无失败下完成，随后补入取消请求的保护和测试。
