# 小样本 SFT 的完整 RD Test100 留出对照

日期：2026-09-14。应用户要求，在 Dev100 对照完成后，对同一次 SFT 保存的
before/after sampler 做完整 Test100 比较。使用 `sft_smoke_v2_monitored`：
Qwen3.5-4B、合成 Train8、16 步 LoRA、rank 8、peak LR 5e-5、10% warmup。
没有依据本次 Test 结果选择 checkpoint、重分数据或训练模型，也没有 W&B 上传。

## 固定协议

- 完整冻结 Test100，两阶段都计算 NLL 并自由生成全部 100 条；错误/截断输出不删除。
- 使用同一轮初始化的 before sampler，不把历史 Dev baseline 当成 Test baseline。
- 与 [RD Dev100 对照](RD_DEV100_SFT_RESULTS.md) 相同的 processor、renderer、
  图片上限 1,048,576 pixels、greedy、seed 20260913、最大生成 8,192 tokens，
  同版本官方 RD 评分器和补充指标。
- NLL 对原始参考 HTML 的 assistant tokens 与结束 token 计算；完整参考序列最长
  12,455 tokens，未截断或排除。PPL 为 exp（全局监督 token mean NLL）。
  自由生成的输入不包含参考 HTML。
- Test manifest SHA-256：
  `44b9bb3bbfa2b938612f979fec7a6a863761a3c4f67496a1701b5b6b2b2c1ce2`。
  Test ID 与冻结划分的 Test100 完全一致，路径和文件哈希映射未变；
  与 Synthetic Train8、RD Dev100 的 ID、图片字节、标签字节和解码 RGB 像素交集均为 0；
  精确重复检查不能排除近重复或预训练污染。

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.checkpoint_eval \
  --source-run "$SFT_RUN_DIR/run.json" \
  --manifest "$LAB_WORK_DIR/data/manifests/rd_test.jsonl" \
  --split test --expected-examples 100 \
  --data-root "$LAB_WORK_DIR" --output-dir "$CHECKPOINT_EVAL_DIR" \
  --tinker-cookbook-dir "$TINKER_COOKBOOK_DIR" --official-repo "$RD_OFFICIAL_REPO" \
  --env-file "$ENV_FILE" --execute
```

默认 `--split dev` 保持原行为；Test 必须显式选择。样本数不符或 split 混用会在
模型加载和付费调用之前报错。不带 `--execute` 仅做本地预检查。

## 结果

**平均内容质量与输出可靠性有改善，但不是所有样本或指标都改善。** Cell F1 和
格式门控 RD 的配对增益区间在本次抽样假设下高于 0；原版 RD 的增益区间跨 0，
尚不足以断言该指标有明确提升。整表完全一致率反而下降，不能只呈现平均值。

| 指标 | Base（before） | SFT 后 | 变化 |
| --- | ---: | ---: | ---: |
| Assistant NLL ↓ | 0.143115 | 0.125075 | −0.018041 |
| Perplexity ↓ | 1.153863 | 1.133233 | −0.020630 |
| 官方原版 RD similarity ↑ | 0.820738 | 0.831475 | +0.010736 |
| RD similarity，格式失败记 0 ↑ | 0.744963 | 0.826470 | +0.081507 |
| Cell F1 ↑ | 0.407156 | 0.497065 | +0.089908 |
| Numeric F1 ↑ | 0.444528 | 0.616441 | +0.171913 |
| 格式通过率 ↑ | 89/100 | 99/100 | +10 张 |
| 输出截断率 ↓ | 5/100 | 1/100 | −4 张 |
| 整表完全一致率 ↑ | 7/100 | 3/100 | **−4 张** |
| 结构完全一致率 ↑ | 14/100 | 22/100 | +8 张 |

Numeric F1 两阶段各有 98 个有效样本，其余按现有规则不参与该指标均值；全部
100 条仍参与整体评测；98 个有效 Numeric F1 样本的 ID 前后相同。
整表完全一致要求维度、标准化单元格内容及 spans 全部匹配。

| 配对指标（100 对） | 平均差 | 配对 bootstrap 95% 区间 | 改善 / 持平 / 退步 |
| --- | ---: | --- | --- |
| Cell F1 | +0.08991 | [+0.03899, +0.14072] | 51 / 22 / 27 |
| RD，格式失败记 0 | +0.08151 | [+0.03717, +0.12916] | 49 / 21 / 30 |
| 原版 RD | +0.01074 | [−0.01531, +0.03722] | 44 / 20 / 36 |

Cell F1 提升约 **9.0 个百分点**、Numeric F1 提升约 **17.2 个百分点**，说明收益
不限于 teacher-forced NLL。格式有效性和截断改善也会影响内容指标，因此还不能
把全部增益解释为视觉识别变强。仍有 27 张的 Cell F1 下降，且整表完全一致率下降，
支持“平均更好、有退步”的结论，不支持逐样本全面优于 base。

进一步按已有指标核对：原先 7 张完全一致的表格中，3 张保持、4 张退步，没有
新增完全一致；这 4 张都有 cell F1 下降，其中 2 张另有结构变化，均非格式失败或
截断。两阶段都格式有效的 89 张，平均 cell F1 仍提高约 **0.06816**，所以收益
不只是格式失败记零造成的。此为事后分组诊断，不是隔离视觉识别、序列化等因素的
因果实验。独立复核也按固定清单顺序精确复现了三个配对 bootstrap 区间。

两阶段均完成 100 次 NLL 和 100 次自由生成；每阶段 **108,566 个监督 tokens**，
前后 target tokens 完全一致。全量保存的 logprobs 独立复算 NLL/PPL、全部 200 份
预测重新评分、逐样本 ID 对齐及 manifest/source-run 哈希核验均通过。
配对统计的三个指标各有完整 100 对有限值。

总耗时 **573.7 秒（9.6 分钟）**。NLL 完整输入共 344,864 tokens，生成输入
127,732 tokens、输出 267,590 tokens；计入概率 API 额外生成的 200 tokens，
按当前记录的费率估算计算费 **$0.42509**，非供应商账单。预检查计算上界 $1.80275。
没有新增训练，也没有 W&B 上传。仓库测试 **202 passed / 3 skipped**。

## 配对比较方法与边界

以同一张表格的 after − before 为差值。Cell F1 是主要内容指标，格式失败记 0 的
RD similarity 是补充指标；同时报告原版 RD 分数。每个指标报告均值差、改善/
持平/退步样本数（绝对差 ≤ 1e-12 视为持平），并按完整表格配对 bootstrap
10,000 次，固定 seed 20260914，取均值差分布的 2.5%/97.5% 分位数。
不分别重采样两套模型，也不把同一张表格内的 tokens 或 cells 当成独立样本。

这些区间假设表格可以独立抽样；尚未按未知的文档来源聚类，可能低估相关样本带来的
不确定性。它们只描述本次模型和生成结果下的样本不确定性，不覆盖训练随机种子、
服务推理波动或新的数据分布；不是多次训练复现，也没有多指标同时覆盖率保证。

Test100 至此已被查看，不能继续称为“从未使用过的测试集”。按用户随后明确的约定，
每轮迭代都在固定全量 Test100 上评估，与固定 Base 比较；训练期间仍以全量 Dev100
监控。Test100 因而是持续使用的回归比较基准。若未来需要新的独立最终结论，
应另留未经查看且与训练来源隔离的评测集。

## W&B Base 登记

后续单独将本次 **before/Base** 汇总上传到
[qwen35-4b-base-test100-9409ea](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/9409ea84393aebb0)。
group：`rd-test100-0ddf5237b84f`；baseline run ID：`9409ea84393aebb0`。
上传复用了现有预测，没有再次调用推理或训练 API；本页前面的“没有 W&B 上传”
指原始推理运行阶段。此次没有上传 SFT after，其结果仍保留在本地报告。

已从远端读回确认 `finished`、16 项指标及全部受控 config 均一致，远端文件仅
`config.yaml` 和 `wandb-summary.json`。没有图片、HTML、逐样本记录或私有路径。
登记方法及每轮新版本比较规则见 [评测流程](EVALUATION.md#固定-test100-baseline-与每轮迭代登记)。

原始证据只保留在本地 `outputs/rd_test100_sft_v2/`：运行记录、两阶段逐 token
概率、预测与逐样本评分，以及 `verification.json` 和 `paired_comparison.json`。
本报告仅提交汇总，不提交图片、HTML、样本 ID、token IDs、私有路径或 sampler 地址。
