# 澜图智能企业 RAG 评测报告

- 语料命名空间：`lantu`
- 文档数：134，文本块数：603
- 评测用例：74 条（检索类 66 条，拒答类 8 条）
- 生成方式：{'extractive': 70, 'evidence_check': 4}，向量模型：lantu-hash-384
- 召回请求块数：10（下列 TopN 按前 N 个检索块计算）
- 本报告是固定题集的检索/关键词/引用结构回归；独立 reasoning 评测用于核对多跳、计算和时间推断。

## 评分边界

- keyword_accuracy（兼容旧键 answer_accuracy）仅检查标准化关键词是否全部出现，不是语义准确率，不能排除否定、矛盾或错误推理。
- any_expected_doc_hit@5/@10（兼容旧键 recall@5/@10）检查前 N 个检索块是否含任一期望文档，不是全部证据召回率；文档编号在这 N 块内去重。
- all_expected_docs_recalled@10 与 all_expected_citation_coverage 只针对题集标注的期望文档；标注完整性和逐句证据支持仍需人工审查。
- 引用编号校验只验证答案标记、返回引用和召回文档之间的结构一致性，不是语义蕴含或引用忠实度校验。
- 本评测用于原固定题集回归；计数、数值/时间运算、多跳依赖及证据缺失应查看独立 reasoning 评测，不能由本报告的关键词分数推定通过。

## 一、总体指标

| 指标 | 数值 |
| --- | --- |
| 前5块含任一期望文档（旧 Recall@5） | 93.94% |
| 前10块含任一期望文档（旧 Recall@10） | 96.97% |
| 前10块包含全部期望文档 | 93.94% |
| MRR | 0.8254 |
| nDCG@10 | 0.8464 |
| 全部关键词命中率（旧 answer_accuracy） | 90.91% |
| 至少一个期望来源被有效标记引用 | 84.85% |
| 全部期望来源被有效标记引用 | 72.73% |
| 引用编号结构合法率 | 93.94% |
| 系统 grounded 标志率（非独立语义审查） | 93.94% |
| 拒答准确率 | 100.00% |
| 非拒答问题误拒率 | 6.06% |
| 平均延迟（毫秒） | 17.0 |

## 二、分类指标

| 类别 | 用例数 | 任一期望文档@5 | 任一期望文档@10 | MRR | 全关键词命中 | 至少一期望来源引用 | 拒答准确率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_doc | 16 | 100.00% | 100.00% | 0.8542 | 93.75% | 93.75% | 0.00% |
| multi_doc | 10 | 100.00% | 100.00% | 0.8250 | 90.00% | 80.00% | 0.00% |
| cross_department | 8 | 100.00% | 100.00% | 0.8167 | 87.50% | 75.00% | 0.00% |
| conflict | 8 | 100.00% | 100.00% | 1.0000 | 87.50% | 87.50% | 0.00% |
| time_range | 8 | 100.00% | 100.00% | 1.0000 | 100.00% | 100.00% | 0.00% |
| fuzzy | 8 | 87.50% | 100.00% | 0.7639 | 100.00% | 87.50% | 0.00% |
| cross_format | 8 | 62.50% | 75.00% | 0.4896 | 75.00% | 62.50% | 0.00% |
| refusal | 8 | — | — | — | — | — | 100.00% |

## 三、失败与拒答明细

| 用例 | 类别 | 问题 | 命中文档 | 缺失关键词 | 是否拒答 |
| --- | --- | --- | --- | --- | --- |
| S14 | single_doc | 澜图智能是一家什么公司？ | lantu_company_profile, lantu_company_mission | 工业机器人、智能制造 | 是 |
| M01 | multi_doc | 2025年公司研发投入和研发人员数量分别是多少？ | lantu_finance_investor_qa, lantu_investor_overview | 5.87、1,286 | 是 |
| X07 | cross_department | 海外项目需要遵守哪些出口管制和数据合规要求？ | lantu_org_committees, lantu_legal_compliance_manual | 两用物项、GDPR | 是 |
| C08 | conflict | 2025版销售折扣制度规定的项目毛利率底线是多少？ | lantu_sales_discount_2025, lantu_finance_annual_2025 | 28% | 是 |
| CF03 | cross_format | 2025年收入构成中机器人本体占比是多少？ | lantu_product_r_series, lantu_product_matrix | 46.5% | 否 |
| CF05 | cross_format | 2025年末员工总数和苏州基地人数分别是多少？ | lantu_org_chart_2025, lantu_meeting_safety | 2,450 | 否 |

## 四、逐条结果

| 用例 | 类别 | 问题 | Top1 文档 | 首个相关排名 | 关键词/拒答检查通过 | 拒答 | 耗时(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S01 | single_doc | 澜图智能成立于哪一年，总部在哪里？ | lantu_company_profile | 1 | 是 | 否 | 780 |
| S02 | single_doc | 公司2025年机器人出货量和累计交付量是多少？ | lantu_company_profile | 1 | 是 | 否 | 0 |
| S03 | single_doc | LT-R20 的负载、臂展和重复定位精度是多少？ | lantu_product_r20_pro_whitepaper | 3 | 是 | 否 | 15 |
| S04 | single_doc | LT-C800 控制器支持几轴联动，EtherCAT 总线周期是多少？ | lantu_product_controller | 1 | 是 | 否 | 0 |
| S05 | single_doc | 员工年假按工龄分别是多少天？ | lantu_policy_employee_handbook | 1 | 是 | 否 | 16 |
| S06 | single_doc | 2025年7月版差旅标准中，一线城市住宿和餐饮补贴是多少？ | lantu_policy_travel_2025h2 | 1 | 是 | 否 | 0 |
| S07 | single_doc | 2025年公司营业收入和归母净利润是多少？ | lantu_finance_annual_2025 | 1 | 是 | 否 | 16 |
| S08 | single_doc | P1 故障的响应时间和到场时间分别是多久？ | lantu_support_sla_2025 | 1 | 是 | 否 | 0 |
| S09 | single_doc | LT-R20 的减速器 RV-40E 备件价格是多少？ | lantu_support_spare_parts_price | 1 | 是 | 否 | 14 |
| S10 | single_doc | 销售合同标准模板的预付款比例和质保期是多少？ | lantu_legal_sales_contract_template | 1 | 是 | 否 | 0 |
| S11 | single_doc | 员工账号密码的更换周期是多久？ | lantu_policy_infosec | 1 | 是 | 否 | 0 |
| S12 | single_doc | 2025年华南大区销售完成率是多少？ | lantu_meeting_sales | 2 | 是 | 否 | 16 |
| S13 | single_doc | 公司叫什么名字？ | lantu_policy_employee_handbook | 2 | 是 | 否 | 0 |
| S14 | single_doc | 澜图智能是一家什么公司？ | lantu_company_profile | 1 | 否 | 是 | 16 |
| S15 | single_doc | 公司总部在哪里？ | lantu_company_campus | 3 | 是 | 否 | 0 |
| S16 | single_doc | 公司的使命是什么？ | lantu_company_mission | 1 | 是 | 否 | 0 |
| M01 | multi_doc | 2025年公司研发投入和研发人员数量分别是多少？ | lantu_finance_investor_qa | 4 | 否 | 是 | 15 |
| M02 | multi_doc | 恒芯动力电池项目使用了哪些机器人，项目成效如何？ | lantu_meeting_case_review | 1 | 是 | 否 | 0 |
| M03 | multi_doc | 锂电叠片方案的良率目标和标杆客户是谁？ | lantu_solution_lithium | 1 | 是 | 否 | 15 |
| M04 | multi_doc | 半导体晶圆搬运方案的洁净等级和标杆客户是谁？ | lantu_solution_semiconductor | 1 | 是 | 否 | 0 |
| M05 | multi_doc | 2025年海外营业收入是多少，欧洲技术中心设在哪里？ | lantu_company_history | 2 | 是 | 否 | 15 |
| M06 | multi_doc | 售后服务团队有多少人，全国服务网点多少个？ | lantu_support_system | 1 | 是 | 否 | 0 |
| M07 | multi_doc | 2026年营业收入目标、研发投入预算和资本开支分别是多少？ | lantu_finance_budget_2026 | 1 | 是 | 否 | 16 |
| M08 | multi_doc | 销售折扣 15%-25% 由谁审批，项目毛利率底线是多少？ | lantu_sales_discount_2025 | 1 | 是 | 否 | 0 |
| M09 | multi_doc | LT-R20 Pro 计划何时量产，额定负载是多少？ | lantu_product_r20_pro_whitepaper | 1 | 是 | 否 | 16 |
| M10 | multi_doc | 2025年客户投诉目标是多少起，一次交检合格率目标是多少？ | lantu_meeting_quality | 2 | 是 | 否 | 0 |
| X01 | cross_department | 研发人员占员工总数比例和研发投入占营业收入比例分别是多少？ | lantu_company_profile | 3 | 是 | 否 | 0 |
| X02 | cross_department | 新员工入职培训需要几天，试用期一般多长？ | lantu_hr_new_hire_training | 1 | 是 | 否 | 14 |
| X03 | cross_department | 员工出差一线城市住宿上限和费用报销时限分别是多少？ | lantu_policy_travel_2025h2 | 1 | 是 | 否 | 0 |
| X04 | cross_department | 采购标准供应商的付款账期和销售合同预付款比例分别是多少？ | lantu_policy_purchase_2025 | 1 | 是 | 否 | 16 |
| X05 | cross_department | 公司数据分为几个等级，L3 数据访问需要谁审批？ | lantu_policy_data_classification | 1 | 是 | 否 | 0 |
| X06 | cross_department | 2025年人均培训学时和2026届校招计划人数是多少？ | lantu_hr_campus | 1 | 是 | 否 | 16 |
| X07 | cross_department | 海外项目需要遵守哪些出口管制和数据合规要求？ | lantu_org_committees | 5 | 否 | 是 | 0 |
| X08 | cross_department | 新员工安全培训不少于多少小时，2025年安全投入是多少？ | lantu_meeting_safety | 1 | 是 | 否 | 15 |
| C01 | conflict | 一线城市出差住宿标准是多少？ | lantu_policy_travel_2025h2 | 1 | 是 | 否 | 0 |
| C02 | conflict | 销售折扣审批权限如何划分？ | lantu_sales_discount_2025 | 1 | 是 | 否 | 15 |
| C03 | conflict | 售后服务 P1 的响应时效是多久？ | lantu_support_sla_2025 | 1 | 是 | 否 | 0 |
| C04 | conflict | 公司研发中心下设几个研究所？ | lantu_org_chart_2025 | 1 | 是 | 否 | 15 |
| C05 | conflict | 员工每月可以申请远程办公几天？ | lantu_policy_attendance_2025 | 1 | 是 | 否 | 0 |
| C06 | conflict | 采购标准供应商的付款账期是多少天？ | lantu_policy_purchase_2025 | 1 | 是 | 否 | 0 |
| C07 | conflict | 销售折扣超过 25% 由谁审批？ | lantu_sales_discount_2025 | 1 | 是 | 否 | 16 |
| C08 | conflict | 2025版销售折扣制度规定的项目毛利率底线是多少？ | lantu_sales_discount_2025 | 1 | 否 | 是 | 0 |
| T01 | time_range | 2024年一线城市出差住宿标准是多少？ | lantu_policy_travel_2024 | 1 | 是 | 否 | 16 |
| T02 | time_range | 2025年1月版差旅标准中一线城市住宿上限是多少？ | lantu_policy_travel_2025 | 1 | 是 | 否 | 0 |
| T03 | time_range | 2025年7月修订后二线城市住宿标准是多少？ | lantu_policy_travel_2025h2 | 1 | 是 | 否 | 0 |
| T04 | time_range | 2024年公司营业收入和归母净利润是多少？ | lantu_finance_annual_2024 | 1 | 是 | 否 | 14 |
| T05 | time_range | 2025年第三季度单季营业收入是多少？ | lantu_finance_q3_2025 | 1 | 是 | 否 | 0 |
| T06 | time_range | 2024年研发投入占营业收入比例是多少？ | lantu_finance_annual_2024 | 1 | 是 | 否 | 16 |
| T07 | time_range | 2025年上半年新签订单金额是多少？ | lantu_finance_h1_2025 | 1 | 是 | 否 | 0 |
| T08 | time_range | 2024年末公司员工总数是多少？ | lantu_org_chart_2024 | 1 | 是 | 否 | 16 |
| F01 | fuzzy | 出差住宿报销上限怎么算？ | lantu_policy_travel_2025h2 | 1 | 是 | 否 | 0 |
| F02 | fuzzy | 机器人本体保修多久？ | lantu_product_r_series | 2 | 是 | 否 | 0 |
| F03 | fuzzy | 公司去年赚了多少钱？ | lantu_finance_q3_2025 | 9 | 是 | 否 | 15 |
| F04 | fuzzy | 售后工程师多久能到现场处理停机？ | lantu_support_system | 2 | 是 | 否 | 0 |
| F05 | fuzzy | 买机器人最多能打几折，找谁批？ | lantu_sales_discount_2025 | 1 | 是 | 否 | 15 |
| F06 | fuzzy | 新来的员工要培训几天？ | lantu_hr_new_hire_training | 1 | 是 | 否 | 0 |
| F07 | fuzzy | 公司有哪些移动机器人产品？ | lantu_product_amr | 1 | 是 | 否 | 15 |
| F08 | fuzzy | 客户投诉怎么处理，多久闭环？ | lantu_support_complaint | 1 | 是 | 否 | 0 |
| CF01 | cross_format | 2025年12月的营业收入是多少？ | lantu_company_profile | 3 | 是 | 否 | 16 |
| CF02 | cross_format | LT-AMR-1000 的负载和续航分别是多少？ | lantu_product_amr | 1 | 是 | 否 | 0 |
| CF03 | cross_format | 2025年收入构成中机器人本体占比是多少？ | lantu_product_r_series | — | 否 | 否 | 0 |
| CF04 | cross_format | LT-AMR-1000 锂电池组备件的价格和质保期是多少？ | lantu_support_spare_parts_price | 1 | 是 | 否 | 0 |
| CF05 | cross_format | 2025年末员工总数和苏州基地人数分别是多少？ | lantu_org_chart_2025 | — | 否 | 否 | 0 |
| CF06 | cross_format | 2025年第三季度的订单金额是多少？ | lantu_finance_q3_2025 | 4 | 是 | 否 | 14 |
| CF07 | cross_format | 2025年公司毛利率和经营活动现金流净额是多少？ | lantu_investor_overview | 3 | 是 | 否 | 0 |
| CF08 | cross_format | 2025年4月董事会通过的2024年度利润分配方案是什么？ | lantu_meeting_board | 1 | 是 | 否 | 16 |
| R01 | refusal | 公司食堂今天中午的菜单是什么？ | lantu_policy_travel_2025h2 | — | 是 | 是 | 16 |
| R02 | refusal | 创始人顾承澜的生日是哪一天？ | lantu_org_board | — | 是 | 是 | 0 |
| R03 | refusal | 澜图智能在印度的工厂地址是什么？ | lantu_company_mission | — | 是 | 是 | 0 |
| R04 | refusal | 公司是否提供员工宠物保险？ | lantu_legal_purchase_contract_template | — | 是 | 是 | 15 |
| R05 | refusal | 2026年世界杯冠军是谁？ | lantu_tech_roadmap | — | 是 | 是 | 0 |
| R06 | refusal | 请推荐适合澜图智能的并购标的并给出估值。 | lantu_hr_social | — | 是 | 是 | 15 |
| R07 | refusal | 公司股票明天的收盘价会是多少？ | lantu_investor_overview | — | 是 | 是 | 0 |
| R08 | refusal | LT-R20 的 Python SDK 源码在哪里下载？ | lantu_meeting_product_review | — | 是 | 是 | 0 |

完整机器可读逐条结果见 `per_case.jsonl`，包含检索 Top10、实际答案/生成方法、有效与无效引用编号、全部期望来源完整度、缺失关键词与改写信息。

## 五、结论与分析

- 本次向量模型、实际答案生成方法和检索参数以上方配置及逐条结果为准；hash 向量仅表示词项/字符特征相似，不等于语义模型。
- 全关键词命中可能与错误结论同时出现；必须结合实际回答与独立 reasoning 评测判断推理能力。
- 全部期望文档召回与引用完整度揭示多来源缺失；合法引用编号仍不能证明每个事实被资料支持。
- 原 gate 阈值保留用于固定题集回归，不能用通过 gate 替代语义审查或多跳推理验收。