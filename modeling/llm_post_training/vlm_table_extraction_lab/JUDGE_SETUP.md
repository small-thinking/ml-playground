# Table Judge 本地 setup

2026-09-13 已安装官方
[reductoai/table-judge-benchmark](https://github.com/reductoai/table-judge-benchmark)，
版本固定为 `5e10ee1af6696a978b990bd441eb335aa92ea3c7`。

源码路径：`data/tools/table-judge-benchmark`，稀疏取出 src/scripts 及根目录
文件。依赖按官方 `uv.lock` 安装到它自己的 `.venv`（Python 3.12.7），没有
修改 ML Playground 的 pyproject、lockfile 或环境。

## 已实际验证

- 官方 `load_manifest` 对本地 538 个案例和 1,614 个 SHA-256 的校验通过。
- 官方 `table-judge-run --dry-run` 生成一模型 1,076 次潜在调用的计划；
  实际 API 调用数为 0。`gpt-5.4` 仅作为可识别 provider 名字用于 dry-run，
  并非已选定的 judge 或获准调用的付费模型。
- 官方三维输出解析器：正确布尔标签通过，缺失/非法/矛盾标签被拒绝。
- 官方分析器：手工构造的 oracle、全部接受、全部拒绝软件样例分别得到
  预期 FPR/TPR = 0/1、0/0、1/1。这是软件校验，不是任何模型的评测成绩。
- 结果保存于 `outputs/judge_setup/verification.json`、
  `official_dry_run.json`、`software_fixtures_only.*`。

运行本地无 API 检查：

```bash
uv run --no-sync python -m modeling.llm_post_training.vlm_table_extraction_lab.judge_preflight --work-dir "$LAB_WORK_DIR" --official-repo "$TABLE_JUDGE_REPO"
```

## 支持边界

上游 `call_model` 内置 OpenAI、Anthropic 和 Gemini 接口；本次只安装默认
依赖，Anthropic/Gemini 的可选 SDK 尚未安装。没有本地 Qwen/Tinker adapter。
选定 judge 后才需要增加对应的输入/生成适配；官方 prompt、输出解析器和
分析器可以保持一致。不要把一个 Qwen 模型名直接传给上游 runner 并认为
它已经支持本地 GPU 或 OpenAI-compatible 服务。

官方 prompt 包含三个判定：content_accuracy、structural_preservation、
formatting_fidelity。主报告同时看 FPR、TPR、注入错误对应 rubric 的检出率、
coverage 和各错误类别；不能把解析失败静默丢弃后只报告高分。

## 与抽取评测的关系

这个工具检查给定 HTML 是否忠实于图片。我们的学生任务是图片 → HTML。
两者需要不同的模型输入和结果记录。抽取任务主指标仍应由学生输出与 GT
的内容/数字/结构对比得到；judge 作为经过校准的辅助信号。

Judge 的 clean 图是从 HTML 渲染得到，因此 clean verdict 有构造依据。
这解释了它适合测试 judge 误报；也意味着用它作抽取 benchmark 时要注明
“渲染表格分布”，不能直接等同于真实扫描件表现。
