# Evaluation v1

本阶段评测“表格图片 → HTML”抽取任务。无需训练；也不要求启动推理服务。
可读取已经生成的预测 JSONL，或用 Transformers 在当前机器直接推理。
Qwen3.5-4B 自带视觉能力，正式模型 ID 是 `Qwen/Qwen3.5-4B`。
Tinker 也能提供推理，但本 PR 不依赖其 SDK；以后只需导出相同预测格式。

## 安装与官方代码

在仓库根目录执行（Python 3.11+ 为当前验证环境；Qwen3.5 需支持该架构的
Transformers 版本，仓库锁文件在 Python 3.10+ 上解析到 5.8.1）：

```bash
uv sync --locked --extra dev --extra table-eval
```

不重新分发上游实现；把 [RD 官方仓库](https://github.com/reductoai/rd-tablebench)
克隆到自己的本地目录，固定 revision：

```bash
git clone https://github.com/reductoai/rd-tablebench.git "$RD_SCORER_DIR"
git -C "$RD_SCORER_DIR" checkout 1cae108e6395ddc8389af17385f9769519070558
```

`official.py` 在执行之前验证 `grading.py` 和 `convert.py` 的固定 SHA-256。
不自动下载或执行新版本。采用上游 HTML 转换与 `table_similarity` 原始逻辑，
不更改其归一化或分数；解析失败的预测计零。过大表格触发计算上限时，官方
分数记为不可用并增加 `official_error`，不能将其冒充完整覆盖的 benchmark。

## 输入与执行

先在自己的 shell 中设置以下变量，真实路径不进入代码或 Git：
`EVAL_MANIFEST`、`DATA_ROOT`、`PREDICTIONS_FILE`、`RD_SCORER_DIR`、`EVAL_OUTPUT_DIR`。
Manifest 每行一个 JSON 对象，字段如下（下面的名字只是合成示例）：

```json
{"id":"sample-001","image":"images/sample-001.jpg","label":"labels/sample-001.html","split":"dev"}
```

`image` 和 `label` 相对 `--data-root` 解析，也支持调用者明确提供的绝对路径。
生成预测时只读取图片和固定指令，不把标签传入模型。
可提供 `image_sha256`、`label_sha256`，评测会核验相应输入；数据准备脚本会生成它们。
已有 RD 清单可以直接传入，其中相对路径以准备数据时的 `--work-dir` 为根。
Manifest 决定精确评测集合；开发阶段传 Dev，Test 只在最后固定模型时传入。

预测 JSONL：

```json
{"id":"sample-001","html":"<table><tr><td>-100</td></tr></table>","stop_reason":"stop","input_tokens":100,"output_tokens":20,"latency_seconds":0.5}
```

`id`、`html` 为内容字段，其他 telemetry 可选；`cost_usd` 也可选。没有遥测时
报告 count=0，不把未知费用或截断率当作零。模型/tokenization/采样设置应由
预测生产方保存；本评测不能从 HTML 猜出这些配置，也不能证明外部预测未见标签。
重复 ID、额外 ID、无效参考标签会报错；缺失预测保留在分母中并计为失败。

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.evaluate \
  --manifest "$EVAL_MANIFEST" --data-root "$DATA_ROOT" \
  --predictions "$PREDICTIONS_FILE" --official-repo "$RD_SCORER_DIR" \
  --output-dir "$EVAL_OUTPUT_DIR" --wandb-project vlm-table-extraction
```

默认 W&B online。通过环境提供 `WANDB_API_KEY`，或显式传 `--env-file "$ENV_FILE"`。
离线开发用 `--wandb-mode offline`，不记录则用 `--wandb-mode disabled`。
离线 run 保存在 output 下的 `wandb_offline`；需要时可用 `wandb sync` 同步其中
具体 offline run 目录。W&B 失败会非零退出，本地结果保留为 pending，避免假报成功。

本机或租用 GPU 上直接推理：

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.evaluate \
  --backend transformers --model Qwen/Qwen3.5-4B --revision "$MODEL_REVISION" \
  --device cuda --max-new-tokens 4096 --max-pixels 1048576 \
  --manifest "$EVAL_MANIFEST" --data-root "$DATA_ROOT" \
  --official-repo "$RD_SCORER_DIR" --output-dir "$EVAL_OUTPUT_DIR" \
  --wandb-project vlm-table-extraction
```

也接受 `--model "$LOCAL_MODEL_DIR"`；本地路径只进本地 provenance。
模型首次运行可能下载权重；本 PR 没有下载或实际运行 4B，GPU/MPS 的显存与
吞吐尚未测量。`--device cpu` 是安全默认值，不是吞吐推荐。
逐张推理，关闭 thinking，greedy decoding，按像素上限等比缩小（processor
可能进一步调整尺寸）。保存模型 revision 和实际 token 数；正式对照需固定
模型、processor、图片处理和生成配置。遇推理错误立即退出，已完成预测逐行
落盘，可用预测文件模式评估已完成部分；它会把其余缺失预测计为失败。

## 指标定义

所有质量分数范围 0–1，按样本宏平均并记录各自有效样本数。

| 指标 | 定义与边界 |
| --- | --- |
| `official_rd_similarity` | 原版 RD similarity；不是“正确单元格百分比”。会去掉负号并宽容边界缺失 |
| `cell_precision/recall/f1` | 展开网格后，按位置和规范化文本精确匹配；缺失/新增位置影响召回/精度 |
| `cell_bag_f1` | 忽略位置的单元格多重集 F1，仅作诊断；重复值有计数 |
| `numeric_precision/recall/f1` | 按单元格位置、数字出现序号、数字 token 精确匹配；保留符号、括号、百分号和分隔符，不猜 locale |
| `row_count_exact/column_count_exact` | 完整行数与最大逻辑列数是否一致 |
| `span_f1/structure_exact` | 按原始单元格起点、rowspan/colspan 比较结构；能区别合并与重复单元格 |
| `table_exact` | 展开后的文本及原始 span 结构全部一致 |
| `parse_success/empty_output` | 恰好一个非空闭合表格、可接受跨度，无外围解释文字；可有 HTML 包装或一个 Markdown fence |
| `prediction_present/truncated` | 预测覆盖率；有 stop reason 时检测 length/max_tokens 截断 |
| tokens / latency / cost | 有数据的均值、总和和 count；wall_seconds 包括当前后端的模型加载及评测，不含 W&B 上传 |

文本做 NFKC、Unicode 负号与空白规范化，保留大小写和标点。解析使用 lxml 的
HTML 恢复机制，所以 parse_success 不代表通过严格 HTML 标准验证。rowspan
超出已有行时按现有行裁剪网格，但保留声明跨度作结构比较；参考数据存在此情况。
数值指标是字符串 token 指标，不等同数学等价判断，也不完整覆盖所有数字写法。
两侧都无数字的样本不参与 numeric 平均，count 明确分母。空/无效预测不会从
其他质量平均中消失。暂不实现 TEDS、样式准确率或 LLM judge 调用。

Table Judge 官方任务是“图片 + 候选 HTML → 判断错误”，与本抽取任务不同。
原先的 `judge_preflight` 保留其离线 setup，不能把它的 smoke 或人工 fixture
指标当作 VLM baseline；后续接入可靠 judge 时应单列费用、解析失败和 judge 校准。

## 输出、隐私与验证

本地生成 `summary.json`、`per_sample.jsonl`、`provenance.json`，直接推理还会
生成 `predictions.jsonl`。使用新 output 目录避免覆盖结果。上述文件、原始数据、
凭证和清单均不进入 PR；Git 忽略这些格式和目录。未来明确要发布的图片须走
目录级 Git LFS 规则；本 PR 没有图片。

W&B 在独立进程中仅收到 allowlist 配置和汇总数值：evaluator/version、官方
revision、输入清单哈希、backend、受控生成配置。不上传逐样本 ID、图片、
标签、HTML、真实路径或完整命令行。关闭 console、code、Git、机器元数据
与系统监控，清理 W&B 自动配置环境；保留认证与 entity/base URL。离线测试
检查实际 SDK 文件和 `.wandb` 数据中的私有路径标记，而不只 mock API。
当前验证 SDK 为 W&B 0.21.1；升级后需继续运行此回归。

```bash
RD_OFFICIAL_REPO="$RD_SCORER_DIR" uv run --no-sync pytest -q \
  tests/modeling/llm_post_training/test_table_evaluation.py
```

测试使用合成表格，不提交图片或样本数据；官方兼容回归需要显式提供本地官方
代码，未提供时只跳过这一项。CI 安装 table-eval extra 并运行其余测试。

## 本 PR 的实测结果

- 全量本地测试和新增表格回归覆盖负号、缺行、合并单元格、无效输出、重复 ID、
  丢失预测、HTML 包装及 W&B 隐私；最终数量以 PR Test Plan 为准。
- 本地固定官方代码的兼容回归通过；Dev 100 条参考 HTML 均可解析，无样本
  在自比较时超过官方评分计算上限。
- 已用一个合成软件 fixture 跑通 CLI 和 W&B online，并读回确认 run finished、
  聚合指标正确；远端文件仅 config.yaml 和 wandb-summary.json。它不是模型结果。
- 未下载 Qwen3.5-4B 权重、未执行真实 4B 推理或任何训练；Transformers 后端
  已验证架构映射和模拟生成协议，硬件可运行性仍需实际 smoke。
