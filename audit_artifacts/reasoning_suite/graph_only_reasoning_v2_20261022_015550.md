# 推理与版本对比规模测试报告（graph_only_reasoning_v2）

- 模型标识：`qwen3.8-flash`
- 服务地址：`http://127.0.0.1:8080`
- 检索模式：`graph`
- 用例总数：**40**，通过 **38**，失败 **2**，通过率 **95.00%**
- 平均耗时：2885.9 ms，总耗时 115.5 s

## 分类通过率

| 类别 | 总数 | 通过 | 通过率 |
| --- | --- | --- | --- |
| multi_hop_relation | 20 | 20 | 100.00% |
| version_comparison | 15 | 15 | 100.00% |
| version_conflict | 5 | 3 | 60.00% |

## 失败分类

| 类型 | 数量 |
| --- | --- |
| unexpected_refusal | 2 |

## 失败明细

| ID | 类别 | 失败类型 | 问题 | 缺失关键词 | 缺失文档 | 回答摘录 |
| --- | --- | --- | --- | --- | --- | --- |
| V01 | version_conflict | unexpected_refusal | 2024年控制系统架构的控制器型号是什么？ | LT-C600 | lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| V05 | version_conflict | unexpected_refusal | 2024年旧版架构和2025年新版架构的联动轴数分别是多少？ | 4 轴、8 轴 | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
