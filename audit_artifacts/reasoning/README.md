# 推理专项验收

本目录保存与企业业务语料隔离的固定合成用例、实际回答、检索/引用证据、代码哈希与回归结果。只替换 `enterprise_rag.retrieval.load_index` 和 `enterprise_rag.store.load_index`，其余检索与生成流程都执行真实代码，没有模型替身，也未将合成文档写入数据库。

| 测试集 | 修复前严格通过 | 修复后严格通过 | 说明 |
| --- | --- | --- | --- |
| 固定 20 例 | 6 / 20 | 20 / 20 | 含四部门、1956、去重、金额/单位、日期边界、12份证据、权限及未来版本 |
| 独立变式 13 例 | 2 / 13 | 13 / 13 | 在运行修复代码前冻结；数值/顺序、同额合同、亿元、无关公司、三跳主管、非法日历等 |

固定基线最初的 R07 仅关键词检查，错误放过了“原文并列”；现要求显式合成关系结论。只对已保存的原始回答重新评分，未重跑改后代码。原始 7/20 分数保存在 `baseline_initial_grading.json/md`，评分修订记录在 `baseline.json` 的 `grading_note`。固定基线执行时六个核心源码哈希始末相同。变式基线在版本过滤的 `store.py` 已开始修改后运行，另外五个核心文件仍是原实现，哈希逐一留存；变式不包含版本测试。

真实模型调用另有独立边界：`baseline_llm.json` 中四例曾请求 LLM，但全部发生连接错误或超时并降级成抽取，实际 `method=extractive`，严格通过 0/4，不能算模型成功。随后 `transport_diagnostic.json` 中同一成立年问题、同一 qwen3.5-plus 接口，关闭 thinking 并限制输出后两次真实调用约 2.6 / 3.2 秒，均答出 1956 且双引用；不传该设置时以 12 秒读取超时结束。这说明原调用的参数/超时路径存在实测问题，不代表模型没有基本运算能力，也不代表生产全链已经由此验收。

修复后 33 条专项用例明确以 `use_llm=False` 验证有限证据算子和拒答行为；其通过证明这些支持范围内的计算与证据链正确，不等于任意文档、任意句式或任意多跳问题都能推断。生产语义综合的真实模型和 HTTP 验证另见总审计报告。

重跑命令（项目根目录）：

```powershell
.\.venv312\Scripts\python.exe -X utf8 scripts\evaluate_reasoning.py --phase regression --gate
.\.venv312\Scripts\python.exe -X utf8 scripts\evaluate_reasoning.py --phase regression --fixtures audit_artifacts\reasoning\holdout_cases.jsonl --output-name holdout_regression --gate
.\.venv312\Scripts\python.exe -X utf8 -m unittest tests.test_reasoning_operators -v
```

`--use-llm` 只代表允许调用模型，仍应逐条查看实际 `method`；确定性算子会优先完成其支持的运算。为避免覆盖证据，使用 `--output-name` 为额外试验选择独立文件名。
