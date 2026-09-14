# 数据准备核验记录

本页记录首次完整性核验；后续 800/100/100 划分、嵌套训练子集及扩大抽样见
[DISTRIBUTION_AND_SPLITS.md](DISTRIBUTION_AND_SPLITS.md)。首次审计的
`split=unassigned` 清单仍作为原始配对清单保留，使用衍生 `rd_train*.jsonl`
和 `rd_dev.jsonl` 加载已分配角色的样本。

完成时间：2026-09-13 20:33 UTC。通过现有 uv 环境实际执行下载、解压和审计。
匿名下载 Judge 时遇到 HF 429，停止限流等待后使用根目录已配置的 HF token
成功续传。未输出密钥、未调用付费模型或租用 GPU。

| 数据集 | 固定 revision | 已核验样本 |
| --- | --- | ---: |
| reducto/rd-tablebench | `7748503e2bd5f210d27aa2ef5fdf4b8aa13099bb` | 1,000 图片/人工 HTML/PDF 配对 |
| reducto/mle-interview | `7222a4c04d8eae8ca13cbca3ae34caed4400239c` | 996 JPEG，无标签 |
| reducto/table-judge-benchmark | `7bf19d636d13c93d7cdf70441ad024d5137db708` | 538 图片/clean HTML/corrupted HTML 案例 |

- 原始仓库文件：1,623 个，共 834,026,330 bytes（不含解压副本和缓存）。
- 全部 2,534 张图像完成解码；ZIP 文件完成解压 CRC 检查。
- RD 图片、PDF、人工 HTML 的 stem 集合完全匹配。
- Judge 上游 manifest 的 1,614 个文件 SHA-256 全部匹配。
- Judge 错误类别：删行 179、数字变化 90、粗斜体变化 179、拼写错误 90。
- 发现 3 组完全重复图片：MLE 内 1 组、Judge 内 2 组，每组 2 个案例。
  RGB 解码像素哈希也确认这 3 组；Judge 的这 2 组 clean HTML 同样重复。
- 三个数据集之间未发现相同文件哈希、RGB 像素哈希或标签文件哈希。
  这不排除缩放、裁剪、重编码后的近重复或同源文档。
- 所有样本仍为 `split=unassigned`，避免在查重和来源分组完成前制造泄漏。

原始数据及逐条清单位于指定 `--work-dir` 下的 `data/`，均被 Git 忽略。
机器可读汇总：`data/manifests/audit.json`；原始文件清单：
`data/manifests/raw_files.json`；重复案例清单：
`data/manifests/exact_duplicates.json`。

验证范围不包含：全部 PDF 的内容渲染、全部人工标签的语义正确性、近重复、
模型兼容性/推理、抽取指标、训练收益。`prepare_data.py --help`、Python
编译检查及 Black 格式检查通过。

首次 MLE 三张查看随后扩展为各数据集 24 张随机抽样，见分布报告。
“面试官持有私有标签”只是可能解释，当前公开材料未证实。
