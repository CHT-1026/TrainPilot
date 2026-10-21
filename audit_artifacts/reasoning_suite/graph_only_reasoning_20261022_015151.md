# 推理与版本对比规模测试报告（graph_only_reasoning）

- 模型标识：`qwen3.8-flash`
- 服务地址：`http://127.0.0.1:8080`
- 检索模式：`graph`
- 用例总数：**40**，通过 **24**，失败 **16**，通过率 **60.00%**
- 平均耗时：1228.5 ms，总耗时 49.1 s

## 分类通过率

| 类别 | 总数 | 通过 | 通过率 |
| --- | --- | --- | --- |
| multi_hop_relation | 20 | 20 | 100.00% |
| version_comparison | 15 | 3 | 20.00% |
| version_conflict | 5 | 1 | 20.00% |

## 失败分类

| 类型 | 数量 |
| --- | --- |
| unexpected_refusal | 15 |
| wrong_answer | 1 |

## 失败明细

| ID | 类别 | 失败类型 | 问题 | 缺失关键词 | 缺失文档 | 回答摘录 |
| --- | --- | --- | --- | --- | --- | --- |
| A01 | version_comparison | wrong_answer | 相比旧架构图，新架构有哪些变化？ | LT-C600、LT-C800 | — | 总线协议从 CANopen 变更为 EtherCAT [1][2]。 总线周期从 2 ms 缩短为 1 ms [1][2]。 架构层级从“工控机 + 运动控制卡”的两层结构变更为感知层、决策层、运动控制层与执行层的四层结构 [1][2]。 旧版单条总线最多连接 16 个从站，新版资料未提及此限制但明确了 EtherCA |
| A03 | version_comparison | unexpected_refusal | 新架构相比旧架构，联动轴数增加了多少？ | 4 轴、8 轴 | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| A04 | version_comparison | unexpected_refusal | 新架构相比旧架构，插补周期缩短了多少？ | 1 ms、0.5 ms | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| A07 | version_comparison | unexpected_refusal | 新架构的伺服采样频率比旧架构提高了多少？ | 8 kHz、16 kHz | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| A08 | version_comparison | unexpected_refusal | 新旧架构的编码器位数分别是多少？ | 17 位、23 位 | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| A09 | version_comparison | unexpected_refusal | 旧架构的实时操作系统是什么，新架构呢？ | Windows、PREEMPT_RT | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| A10 | version_comparison | unexpected_refusal | 新架构的 MTBF 比旧架构提高了多少？ | 30,000、50,000 | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| A11 | version_comparison | unexpected_refusal | 旧架构的工作温度范围是多少，新架构呢？ | 0℃、45℃、-10℃、55℃ | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| A12 | version_comparison | unexpected_refusal | 新旧架构的 EMC 等级分别是多少？ | 3 级、4 级 | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| A13 | version_comparison | unexpected_refusal | 旧架构支持几轴联动，新架构支持几轴？ | 4 轴、8 轴 | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| A14 | version_comparison | unexpected_refusal | 新架构相对旧架构新增了哪些软件能力？ | RRT*、ROS2 | lantu_tech_control_architecture | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| A15 | version_comparison | unexpected_refusal | 旧架构的安全功能有什么不足，新架构如何改进？ | 外部安全继电器、碰撞检测 | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| V01 | version_conflict | unexpected_refusal | 2024年控制系统架构的控制器型号是什么？ | LT-C600 | lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| V03 | version_conflict | unexpected_refusal | 旧版控制系统架构的 MTBF 是多少？ | 30,000 | lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| V04 | version_conflict | unexpected_refusal | 新版控制系统架构的 MTBF 是多少？ | 50,000 | lantu_tech_control_architecture | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
| V05 | version_conflict | unexpected_refusal | 2024年旧版架构和2025年新版架构的联动轴数分别是多少？ | 4 轴、8 轴 | lantu_tech_control_architecture、lantu_tech_control_architecture_v2 | 现有企业资料未覆盖该问题，建议联系对应业务部门确认。 |
