# Evaluation v1

本阶段评测“表格图片 → HTML”抽取任务。无需训练；也不要求启动推理服务。
默认使用 Tinker 托管推理，评分在本地执行。Transformers、Apple Metal/MLX
以及读取已有预测 JSONL 均保留为显式可选后端。
Qwen3.5-4B 自带视觉能力，正式模型 ID 是 `Qwen/Qwen3.5-4B`。
已有 Tinker Dev100 推理实测约 4.7 分钟、估算约 $0.17（非账单）；本次切回
默认后端无需重跑已完成的 baseline。历史结果见 [BASELINE_RESULTS.md](BASELINE_RESULTS.md)。

## 安装与官方代码

在仓库根目录执行（Python 3.11+ 为当前验证环境；Qwen3.5 需支持该架构的
Transformers 版本，仓库锁文件在 Python 3.10+ 上解析到 5.8.1）：

```bash
uv sync --locked --extra dev --extra table-eval --extra tinker
```

不重新分发上游实现；把 [RD 官方仓库](https://github.com/reductoai/rd-tablebench)
克隆到自己的本地目录，固定 revision：

```bash
git clone https://github.com/reductoai/rd-tablebench.git "$RD_SCORER_DIR"
git -C "$RD_SCORER_DIR" checkout 1cae108e6395ddc8389af17385f9769519070558
```

默认 Tinker 后端还需调用者提供干净、固定 commit 的
[Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) 源码目录：

```bash
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git "$TINKER_COOKBOOK_DIR"
git -C "$TINKER_COOKBOOK_DIR" checkout 485726f55d3b2b5abe5fcb4a0d2f3e18e4599dfe
```

通过 `--tinker-cookbook-dir` 显式传入此路径，不 pip 安装 cookbook，不将其源码或
本地数据提交到本仓库。`--revision` 在 Tinker 后端表示 HF processor revision，
默认固定为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，不是托管模型权重 revision。
覆盖此参数时也必须提供完整 40 位 commit SHA，不接受 `main` 或可变 tag。
Tinker 采样 API 未暴露权重 revision，不能把 processor 的固定版本当作远端权重证明。

`official.py` 在执行之前验证 `grading.py` 和 `convert.py` 的固定 SHA-256。
不自动下载或执行新版本。采用上游 HTML 转换与 `table_similarity` 原始逻辑，
不更改其归一化或分数。`official_rd_similarity_raw` 独立于本地格式门槛，
`official_rd_similarity` 则把格式失败计零。过大表格触发计算上限时，官方
分数记为不可用并保留实际计分分母，不能将其冒充完整覆盖的 benchmark。

## 输入与执行

先在自己的 shell 中设置以下变量，真实路径不进入代码或 Git：
`EVAL_MANIFEST`、`DATA_ROOT`、`RD_SCORER_DIR`、`TINKER_COOKBOOK_DIR`、`EVAL_OUTPUT_DIR`。
读取已有预测时再提供 `PREDICTIONS_FILE`；凭证可通过环境或显式 `ENV_FILE` 提供。
Manifest 每行一个 JSON 对象，字段如下（下面的名字只是合成示例）：

```json
{"id":"sample-001","image":"images/sample-001.jpg","label":"labels/sample-001.html","split":"dev"}
```

`image` 和 `label` 相对 `--data-root` 解析，也支持调用者明确提供的绝对路径。
Tinker 生成预测时会把调用者提供的图片和固定指令发送到推理服务，产生采样费用；
参考标签只用于本地评分，不发送给模型。若图片需始终留在本机，请显式选本地后端。
可提供 `image_sha256`、`label_sha256`，评测会核验相应输入；数据准备脚本会生成它们。
已有 RD 清单可以直接传入，其中相对路径以准备数据时的 `--work-dir` 为根。
Manifest 决定精确评测集合；训练期间使用 Dev。按当前实验约定，每轮迭代完成后
还需在固定全量 Test100 上评测，与登记在 W&B 的 Base 比较，见下面的登记流程。

默认推荐的 Tinker 推理与评测命令：

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.evaluate \
  --backend tinker --model Qwen/Qwen3.5-4B \
  --tinker-cookbook-dir "$TINKER_COOKBOOK_DIR" \
  --concurrency 4 --max-new-tokens 8192 --max-pixels 1048576 \
  --manifest "$EVAL_MANIFEST" --data-root "$DATA_ROOT" \
  --official-repo "$RD_SCORER_DIR" --output-dir "$EVAL_OUTPUT_DIR" \
  --wandb-project vlm-table-extraction --env-file "$ENV_FILE"
```

`--backend` 默认 `tinker`，`--concurrency` 默认 4，`--max-new-tokens` 默认 8192；
显式列出便于复现实验。Tinker 需要 `TINKER_API_KEY`，不要求下载完整模型权重
或租 GPU；processor/tokenizer 文件仍由固定 HF revision 加载。关闭 thinking，
保持固定 prompt、图片策略和生成配置后再比较训练前后结果。

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
  --backend predictions --predictions "$PREDICTIONS_FILE" --official-repo "$RD_SCORER_DIR" \
  --output-dir "$EVAL_OUTPUT_DIR" --wandb-project vlm-table-extraction
```

默认 W&B online。通过环境提供 `WANDB_API_KEY`，或显式传 `--env-file "$ENV_FILE"`。
离线开发用 `--wandb-mode offline`，不记录则用 `--wandb-mode disabled`。
离线 run 保存在 output 下的 `wandb_offline`；需要时可用 `wandb sync` 同步其中
具体 offline run 目录。W&B 失败会非零退出，本地结果保留为 pending，避免假报成功。

可选：本机或租用 GPU 上用 Transformers 直接推理：

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.evaluate \
  --backend transformers --model Qwen/Qwen3.5-4B --revision "$MODEL_REVISION" \
  --device cuda --max-new-tokens 8192 --max-pixels 1048576 \
  --manifest "$EVAL_MANIFEST" --data-root "$DATA_ROOT" \
  --official-repo "$RD_SCORER_DIR" --output-dir "$EVAL_OUTPUT_DIR" \
  --wandb-project vlm-table-extraction
```

也接受 `--model "$LOCAL_MODEL_DIR"`；本地路径只进本地 provenance。
模型首次运行可能下载权重。已在 Apple Silicon 上验证原始 Qwen3.5-4B 权重的
Transformers/MPS 和 MLX 本地推理；MLX 的完整 Dev100 结果保留为历史对照。
Transformers 的 `--device auto` 优先 CUDA，其次 Apple MPS，最后 CPU；
MLX 使用 Apple Metal，不将其记录为 CPU 或 CUDA。
逐张推理，关闭 thinking，greedy decoding，按像素上限等比缩小（processor
可能进一步调整尺寸）。保存模型 revision 和实际 token 数；正式对照需固定
模型、processor、图片处理和生成配置。遇推理错误立即退出，已完成预测逐行
落盘，可用预测文件模式评估已完成部分；它会把其余缺失预测计为失败。

## 指标定义

所有质量分数范围 0–1，按样本宏平均并记录各自有效样本数。

| 指标 | 定义与边界 |
| --- | --- |
| `official_rd_similarity_raw` | 直接调用原版 RD similarity；不是“正确单元格百分比”。会去掉负号并宽容边界缺失 |
| `official_rd_similarity` | 本地诊断版本：格式检查不通过时计零，其余调用原版评分 |
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
生成 `predictions.jsonl`。使用新 output 目录避免覆盖结果；中断后可用 `--resume` 继续，必须保持模型、revision、
输入清单、设备、像素和输出上限相同。还会核验本地模型文件的内容指纹或远端
HF 模型的实际 revision；Tinker 则记录服务模型名与固定 processor/cookbook 协议，
不能证明服务端权重在两次运行间完全相同。缺少原始 inference_config.json 时拒绝续跑。已有预测
不会重复生成。上述文件、原始数据、
凭证和清单均不进入 PR；Git 忽略这些格式和目录。未来明确要发布的图片须走
目录级 Git LFS 规则；本 PR 没有图片。

Tinker 最多保留 `--concurrency` 个请求在途。某个请求失败后不再提交新样本；
已发出的其他请求可能仍完成并计费，恢复时以已落盘预测为准。

推理服务与监控服务的边界不同：Tinker 接收图片和固定 prompt，标签仅在本地；
选择本地后端时，模型推理无需向托管服务发送图片。
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
- 已下载固定 revision 的原始 Qwen3.5-4B 权重，Transformers/MPS 与 MLX
  均完成真实本地推理；未启动训练。完整 Dev 结果以对应运行报告为准。
- 已完成 Tinker Dev100 baseline；本次修改默认后端不自动重新调用推理服务。
  既有 baseline 与此次 CLI 变更的验证应分别报告，不能把历史运行称为新 CLI 的实跑。

## 精简的 W&B 默认视图

完整分数、分母、均值和总量仍保存在本地 summary.json。W&B 默认仅显示14项：

| 分组 | 记录的指标 |
| --- | --- |
| quality（4项） | 原版RD直接相似度、位置敏感单元格F1、位置敏感数字F1、整表exact |
| structure（4项） | 行数exact、列数exact、span F1、结构exact |
| runtime（6项） | 格式通过率、截断率、预测覆盖率、平均请求耗时、wall耗时、总tokens |

总样本数、数字指标有效样本数、原版RD实际计分样本数放入config。原先的
格式失败计零RD版本保留在本地，作为格式诊断，不再占用首页主指标。
原版直接评分现在由official.score_raw实现，始终与格式检查分开计算。
原先已记录的旧run保留历史，不因为新的显示方案丢弃历史实验数据。

F1 = 2 × precision × recall / (precision + recall)。例如GT有10个单元格，
预测12个，其中8个在同一行列且内容一致：precision=8/12，recall=8/10，
F1=16/22≈0.727。把数字单独提取并保留位置，就得到数字F1。它们先逐表计算，
再对样本取平均，并非简单汇总所有表格的单元格数量；整体错位会导致很多位置
失配。RD采用较宽松的模糊匹配，因此分数可能远高于这些严格指标。

## 固定 Test100 baseline 与每轮迭代登记

Base 已登记为 [qwen35-4b-base-test100-9409ea](https://wandb.ai/techtao-small-thinking/vlm-table-extraction/runs/9409ea84393aebb0)，
run ID `9409ea84393aebb0`，比较 group 为 `rd-test100-0ddf5237b84f`。
它使用已完成的 Test100 before 预测，重新评分并核验后上传；没有重复付费推理。
Test manifest 哈希及完整 Base/SFT 结果见 [Test100 报告](RD_TEST100_SFT_RESULTS.md)。

后续每轮固定 checkpoint 后，必须用同一个完整 Test100、相同生成协议和评分器评估，
再把新模型汇总作为单独 run 上传，并引用上述 baseline ID。在 W&B 按此 group
过滤，比较 `model_role=base` 与 `model_role=sft` 的相同指标。训练过程仍使用
全量 Dev 监控；Test100 作为持续回归比较基准，不再称为从未查看的最终测试集。

`checkpoint_eval.py --split test` 生成完整 before/after 结果后，登记新模型：

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.log_checkpoint_eval \
  --evaluation-dir "$CHECKPOINT_EVAL_DIR" \
  --manifest "$TEST_MANIFEST" --data-root "$DATA_ROOT" \
  --official-repo "$RD_SCORER_DIR" --stage after \
  --baseline-run-id 9409ea84393aebb0 \
  --wandb-project vlm-table-extraction --env-file "$ENV_FILE" --upload
```

登记 Base 时使用 `--stage before` 并省略 `--baseline-run-id`；当前 Base 已上传，
后续直接复用它。不带 `--upload` 只生成本地可审查 payload。工具核验源训练记录、
manifest、完整预测与 likelihood 覆盖，重算全部生成指标和 NLL/PPL，再通过隔离
telemetry 子进程上传 allowlist；不创建 Tinker 客户端，不重新推理。

相同内容使用确定性的 run ID，重试会继续同一 run；不同预测或源运行使用不同 ID。
group 由数据清单和生成/评分协议哈希决定，改变协议会自动进入新 group，不能直接
混入既有比较。新版本用 `baseline_run_id` 记录对照关系，不覆盖 Base。

checkpoint 登记使用 **16 项**汇总，按 `quality`（5）、`structure`（4）、
`runtime`（5）、`likelihood`（2）分组：在已有视图基础上增加格式门控 RD、NLL/PPL，
不记录未知的单模型 wall time。已有完整对照耗时是两个模型加 NLL 的总时间，不能
拿它冒充 Base 推理耗时；`runtime/total_tokens` 只统计自由生成输入加输出 tokens。
源记录/预测/清单哈希、样本数、有效分母和协议放 config，不上传原始内容或 sampler 地址。
本地 payload/receipt 便于追溯，远端读回需核对 `finished` 状态、全部指标与文件清单。

## Apple Silicon：MLX 本地后端

`--backend mlx` 使用 Apple Metal，要求显式提供已经下载的本地模型目录，
不会调用Tinker或任何托管推理API。当前使用同一份原始BF16 safetensors，
没有量化。Transformers/MPS与MLX的内核和processor版本可能影响输出，
因此把后端、版本和模型revision记录为独立baseline，不混合不同后端的预测。

在独立环境安装，避免改变主训练环境；`MLX_ENV`由调用者指定：

```bash
uv venv "$MLX_ENV" --python 3.11
uv pip install --python "$MLX_ENV/bin/python" \
  mlx-vlm==0.7.0 transformers==5.17.0 wandb==0.21.1 \
  lxml==6.1.3 python-Levenshtein==0.27.5 python-dotenv==1.2.3
HF_HUB_OFFLINE=1 "$MLX_ENV/bin/python" -m modeling.llm_post_training.vlm_table_extraction_lab.evaluate \
  --backend mlx --model "$LOCAL_MODEL_DIR" --revision "$MODEL_REVISION" \
  --model-label Qwen3.5-4B-local-MLX-bf16 --device auto \
  --max-new-tokens 8192 --max-pixels 1048576 \
  --manifest "$EVAL_MANIFEST" --data-root "$DATA_ROOT" \
  --official-repo "$RD_SCORER_DIR" --output-dir "$EVAL_OUTPUT_DIR" \
  --wandb-project vlm-table-extraction --env-file "$ENV_FILE"
```

模型下载通过Hugging Face进行一次，推理时`HF_HUB_OFFLINE=1`阻止再次联网取模型。
W&B仅上传14项分组汇总及受控配置。通过`--model-label`指定公开显示名称，
不要把实际模型路径填入显示名称。默认路线仍为 Tinker；上述 MLX 命令供显式选择
本地计算时使用。现有 Tinker 与 MLX baseline 都保留，不覆盖旧 run。
