"""Source-checked regression tasks. Private values never enter the public task file."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.retrieval_corpus import read_jsonl, write_json, digest

OUT = ROOT / "research/retrieval_repair_20261101"


def main():
    segments = read_jsonl(OUT / "public/segments.jsonl")
    cases, golds = [], []

    def ref(doc, quote):
        matches = [s for s in segments if s["doc_id"] == doc and quote in s["text"]]
        if not matches:
            raise ValueError(f"Unverified source quote: {doc}: {quote}")
        return {"doc_id": doc, "quote": quote, "locator": matches[0]["locator"]}

    def add(n, dimension, question, expected, refs, *, status="answerable", turns=None, after=0):
        qid = f"RET{n:03}"
        prompt = question + "\n仅按以下字段与类型填写answer：" + json.dumps(
            {k: ("number" if type(v) in (int, float) else "boolean" if type(v) is bool else "string|null" if v is None else "array of strings" if isinstance(v, list) else "string")
             for k, v in expected.items()}, ensure_ascii=False) + "。数值单位以题干为准，引用工具返回的原文及locator。"
        case = {"qid": qid, "question": prompt} if turns is None else {"qid": qid, "turns": [turns, prompt]}
        cases.append(case)
        golds.append({"qid": qid, "dimension": dimension, "question": question,
            "source_review": "source quotes checked; deterministic typed regression; independent review pending",
            "retrieval_rubric": {"expected": expected, "status": status,
                                 "accepted_statuses": [status, 'answerable'] if status == 'not_in_corpus' else [status],
                                 "evidence_groups": [[ref(*r)] for r in refs],
                                 "evidence_after_turn": after}})

    org = "lantu_org_reporting_lines"
    matrix = "lantu_product_matrix"
    product = "lantu_product_s_series"
    vision = "lantu_product_vision"
    algo = "lantu_tech_vision"
    history = "lantu_company_history"
    warranty = "lantu_support_warranty"
    training = "lantu_hr_manager_training"
    add(1, "01_精确定位", "沈亦舟的行政直接上级是谁？仅填写姓名。", {"manager": "顾承澜"}, [(org, "顾承澜是沈亦舟的上级")])
    add(2, "01_精确定位", "管理者领导力培训有几个课程模块、共几天？用整数填写modules和days。", {"modules": 6, "days": 12}, [(training, "共 6 个模块 12 天")])
    add(3, "02_同义与别名", "研发资料中的LantuVision视觉软件属于哪个型号系统？该系统的测量精度是多少毫米？填写system和precision_mm（数值）。", {"system": "LT-V3D", "precision_mm": 0.05}, [(vision, "LT-V3D 由结构光相机、视觉控制器与 LantuVision 软件组成"), (vision, "测量精度 0.05 mm")])
    add(4, "02_同义与别名", "公司CTO叫什么名字？其行政直接下属中，研发中心总经理是谁？填写cto和rd_manager。", {"cto": "沈亦舟", "rd_manager": "程知远"}, [(org, "沈亦舟担任 CTO"), (org, "沈亦舟是程知远的上级"), (org, "程知远担任研发中心总经理")])
    add(5, "03_模糊需求定位", "设备超负荷使用损坏还能按标准政策免费保修吗？同时查机器人本体的期限月数和小时数。填写covered、months、hours。", {"covered": False, "months": 24, "hours": 6000}, [(warranty, "超负荷使用"), (warranty, "不在保修范围"), (warranty, "机器人本体保修 24 个月或 6,000 小时")])
    add(6, "03_模糊需求定位", "我想找管理岗培养项目的结业条件：需要案例答辩吗？是否还要360度评估？填写defense和assessment_360。", {"defense": True, "assessment_360": True}, [(training, "结业需通过案例答辩与 360 度评估")])
    add(7, "04_两跳检索", "沈亦舟的直接上级是谁，该上级的完整职务是什么？填写manager、position。", {"manager": "顾承澜", "position": "董事长兼总经理"}, [(org, "顾承澜是沈亦舟的上级"), ("lantu_org_executives", "顾承澜，董事长兼总经理")])
    add(8, "04_两跳检索", "伺服算法组组长的行政直接上级是谁，这位上级的职务是什么？填写manager和position。", {"manager": "周启航", "position": "控制器研究所所长"}, [(org, "林晓担任伺服算法组组长"), (org, "周启航是林晓的上级"), (org, "周启航担任控制器研究所所长")])
    add(9, "05_三跳以上", "从赵一鸣沿行政直接上级逐级向上走三条边，返回包含起点的path数组。", {"path": ["赵一鸣", "林晓", "周启航", "程知远"]}, [(org, "林晓是赵一鸣的上级"), (org, "周启航是林晓的上级"), (org, "程知远是周启航的上级")])
    add(10, "05_三跳以上", "陈璐沿行政直接上级上溯三条边，返回包含起点的path数组。", {"path": ["陈璐", "许博", "唐婧", "罗嘉树"]}, [(org, "许博是陈璐的上级"), (org, "唐婧是许博的上级"), (org, "罗嘉树是唐婧的上级")])
    add(11, "06_汇总去重", "汇报关系文件的研发汇报链加上LT-C900项目成员中，共出现几个不同的人？只统计研发链和该项目负责人/成员，填写count。", {"count": 6}, [(org, "顾承澜是沈亦舟的上级。沈亦舟是程知远的上级。程知远是周启航的上级。周启航是林晓的上级。林晓是赵一鸣的上级"), (org, "周启航同时担任新一代控制器 LT-C900 项目负责人，项目成员包括林晓与赵一鸣")])
    add(12, "06_汇总去重", "以产品参数矩阵为范围，LT-AMR系列有几个不同型号？仅填写count。", {"count": 3}, [(matrix, "A9=LT-AMR-500"), (matrix, "A10=LT-AMR-1000"), (matrix, "A11=LT-AMR-2000")])
    add(13, "07_数值与单位", "LT-S4的额定负载是多少kg、重复精度的正负幅值是多少mm？填写payload_kg、precision_mm（正数）。", {"payload_kg": 4, "precision_mm": 0.02}, [(product, "LT-S4 负载 4 kg、臂展 600 mm、重复精度 ±0.02 mm")])
    add(14, "07_数值与单位", "管理者培训每期名额与全年期数见原文，全年一共多少人次席位？能否保证去重后人数相同？填写seats与unique_guaranteed。", {"seats": 90, "unique_guaranteed": False}, [(training, "每期 30 人，全年 3 期")])
    add(15, "08_时间推断", "截至2026年6月18日，澜图公司从成立日算满多少周年？填写years。", {"years": 15}, [(history, "2011 年 6 月 18 日，公司在苏州工业园区成立")])
    add(16, "08_时间推断", "据发展历程，公司在2016年发布控制器，到2020年推出AMR产品线，相隔多少年？填写years。", {"years": 4}, [(history, "2016 年发布 LT-C600 控制器"), (history, "2020 年推出 AMR 移动机器人产品线")])
    add(17, "09_历史与生效版本", "查2024年5月1日出差适用的一线城市每晚住宿报销上限（元），填写yuan_per_night。", {"yuan_per_night": 450}, [("lantu_policy_travel_2024", "一线城市（北京、上海、广州、深圳）住宿标准 450 元/晚")])
    add(18, "09_历史与生效版本", "查2025年8月1日适用的一线城市每晚住宿报销上限（元），填写yuan_per_night。", {"yuan_per_night": 600}, [("lantu_policy_travel_2025h2", "一线城市（北京、上海、广州、深圳）住宿标准 600 元/晚")])
    add(19, "10_冲突判断", "LT-V3D系统的识别准确率与机器视觉算法平台的缺陷召回率分别是多少百分数？是否同一指标？填写accuracy_percent、recall_percent、same_metric。", {"accuracy_percent": 99.8, "recall_percent": 99.2, "same_metric": False}, [(vision, "识别准确率 99.8%"), (algo, "缺陷检测召回率 99.2%")])
    add(20, "10_冲突判断", "LT-S4产品说明与产品矩阵的重复精度正负幅值是否一致，分别是多少mm？填写manual_mm、matrix_mm、consistent。", {"manual_mm": 0.02, "matrix_mm": 0.02, "consistent": True}, [(product, "重复精度 ±0.02 mm"), (matrix, "D7=±0.02 mm")])
    add(21, "11_多条件筛选", "仅比较LT-S4与LT-S10，依据产品参数矩阵，负载至少8kg且最大速度至少4m/s的有哪些？用models数组按型号字典序返回。", {"models": ["LT-S10"]}, [(matrix, "A7=LT-S4"), (matrix, "A8=LT-S10")])
    add(22, "11_多条件筛选", "仅比较LT-AMR-500/1000/2000，依据参数矩阵，负载至少1000kg且最大速度至少1.5m/s的有哪些？用models数组按型号字典序返回。", {"models": ["LT-AMR-1000"]}, [(matrix, "A9=LT-AMR-500"), (matrix, "A10=LT-AMR-1000"), (matrix, "A11=LT-AMR-2000")])
    add(23, "12_表格与跨表", "查产品矩阵，LT-S10本体重量kg和LT-S系列标准交期周数是多少？填写weight_kg与lead_weeks。", {"weight_kg": 52, "lead_weeks": 4}, [(matrix, "E8=52 kg"), (matrix, "A3=LT-S 系列 | B3=4 周")])
    add(24, "12_表格与跨表", "查备件价目表，LT-R20最贵备件名称、价格元、发货小时数是多少？填写part、yuan、shipping_hours。", {"part": "减速器 RV-40E", "yuan": 12800, "shipping_hours": 48}, [("lantu_support_spare_parts_price", "LT-R20 | 减速器 RV-40E | 12,800 | 48 小时")])
    add(25, "13_跨格式取证", "联合CSV备件表、XLSX产品矩阵、MD保修政策，查询LT-R20最贵备件价格元、机器人负载kg及机器人本体保修月数。填写spare_yuan、payload_kg、warranty_months。", {"spare_yuan": 12800, "payload_kg": 20, "warranty_months": 24}, [("lantu_support_spare_parts_price", "LT-R20 | 减速器 RV-40E | 12,800"), (matrix, "A4=LT-R20 | B4=20 kg"), (warranty, "机器人本体保修 24 个月")])
    add(26, "13_跨格式取证", "联合公司简介PDF与发展历程MD核对成立年份，再查领导力培训DOCX的全年期数。填写founded_year、training_periods。", {"founded_year": 2011, "training_periods": 3}, [("lantu_company_profile", "成立于2011-06-18"), (history, "2011 年 6 月 18 日"), (training, "全年 3 期")])
    add(27, "14_长文与干扰", "查LT-R20 Pro白皮书（不要用LT-R20），额定负载kg、臂展mm分别是多少？填写payload_kg与reach_mm。", {"payload_kg": 22, "reach_mm": 1750}, [("lantu_product_r20_pro_whitepaper", "额定负载 22 kg，臂展 1,750 mm")])
    add(28, "14_长文与干扰", "区别LT-S4与LT-S10，查产品矩阵中LT-S4的最大速度m/s与本体重量kg，填写speed_mps、weight_kg。", {"speed_mps": 4.5, "weight_kg": 38}, [(matrix, "A7=LT-S4"), (matrix, "E7=38 kg | F7=4.5 m/s")])
    add(29, "15_多轮检索更新", "更正：我实际要2024年5月1日的一线城市每晚住宿上限（元），填写yuan_per_night，重新查适用原文。", {"yuan_per_night": 450}, [("lantu_policy_travel_2024", "住宿标准 450 元/晚")], turns="查2025年8月1日适用的一线城市每晚住宿上限。answer仅用yuan_per_night数值字段，单位元。")
    add(30, "15_多轮检索更新", "更正型号为LT-S10，查重复精度正负幅值mm与最大速度m/s，填写precision_mm、speed_mps。", {"precision_mm": 0.02, "speed_mps": 4.0}, [(matrix, "A8=LT-S10"), (matrix, "D8=±0.02 mm"), (matrix, "F8=4.0 m/s")], turns="查产品矩阵LT-S4的重复精度幅值mm和最大速度m/s。answer仅用precision_mm和speed_mps数值字段。")
    add(31, "16_无答案与证据不足", "查2026年5月实际营业收入（亿元），注意区分年度预算与月度实绩。无记录则revenue_yi=null，并填写annual_budget_is_actual。", {"revenue_yi": None, "annual_budget_is_actual": False}, [("lantu_finance_budget_2026", "营业收入目标 52 亿元"), ("lantu_finance_monthly_2025", "澜图智能 2025 年月度经营数据")], status="not_in_corpus")
    add(32, "16_无答案与证据不足", "只按员工汇报关系文件，顾承澜的行政直接上级是谁？文件未记载则manager=null，不将作者或项目负责人当上级。", {"manager": None}, [(org, "顾承澜是沈亦舟的上级")], status="not_in_corpus")
    public = OUT / "cases.jsonl"
    golds[14]['retrieval_rubric']['evidence_groups'][0].append(ref('lantu_company_profile', '成立于2011-06-18'))
    private = OUT / "private/gold.jsonl"
    private.parent.mkdir(exist_ok=True)
    for path, rows in [(public, cases), (private, golds)]:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_json(OUT / "regression_manifest.json", {"cases": len(cases), "source_verified": True,
        "case_sha256": digest(public.read_bytes()), "gold_sha256": digest(private.read_bytes()),
        "purpose": "targeted_regression_not_population_accuracy", "families_not_independent": True,
        "not_claimed_coverage": ["true_long_document_stress", "web", "SQL", "code_search", "production_HTTP_stream"]})
    print(f"prepared {len(cases)} source-checked targeted regression cases; no candidate calls")


if __name__ == "__main__":
    main()
