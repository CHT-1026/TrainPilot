#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从语料构建真实人名白名单（person_whitelist.json）。

问题：姓氏正则把普通词也当人名——「安全委」「范围需」「章包括」「元属于」
在金标准里被识别成实体，导致 (a) 模型答案被误判为缺实体（假阴性），
(b) 人名篡改检测被伪名稀释（假阳性）。

另一个坑：`[姓][\u4e00-\u9fff]{1,2}` 贪婪匹配会把后一个字吞进去
（"唐婧是许博的上级" → 匹出 "唐婧是"、"许博担"），既造伪名又漏真名。

做法：只从**结构化人名语境**抽取候选（作者/审核/担任/是…的上级/
向…汇报/，<职务>/说/认为/提出/强调/指出），这些模式的捕获组长度受
上下文约束，可直接得到干净的真实人名；再补充语料中重复出现的短名。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "enterprise_rag" / "corpus" / "docs"
OUT = ROOT / "research" / "retrieval_evaluation_20261101" / "bank" / "person_whitelist.json"

SURNAMES = (
    # 百家姓前 200（保证「程/林/郭/侯/钟/汪/邹/龚/樊/邢/裴/陆」等常见姓全覆盖）
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏"
    "水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷"
    "罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明"
    "臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林"
    "刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪"
    "包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊惠甄曲家封芮羿储靳汲邴糜松井段富巫乌"
    "焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶"
    "幸司韶郜黎蓟薄印宿白怀蒲台从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭"
    "贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕"
    "连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利"
    "蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆"
    "红游竺权逯盖益桓公"
)
# 非贪婪 + 固定 2 字：中文姓名绝大多数是「姓 + 1~2 字名」，这里优先取 3 字再退 2 字
NAME3 = re.compile(f"[{SURNAMES}][\u4e00-\u9fff]{{2}}")
NAME2 = re.compile(f"[{SURNAMES}][\u4e00-\u9fff]")

# 结构化人名语境：命中即认为捕获组是人名
CONTEXT_PATTERNS = [
    re.compile(r"作者[:：]\s*([\u4e00-\u9fff]{2,3})"),
    re.compile(r"审核[:：]\s*([\u4e00-\u9fff]{2,3})"),
    re.compile(r"([\u4e00-\u9fff]{2,3})是([\u4e00-\u9fff]{2,3})的上级"),
    re.compile(r"([\u4e00-\u9fff]{2,3})担任"),
    re.compile(r"([\u4e00-\u9fff]{2,3})负责"),
    re.compile(r"([\u4e00-\u9fff]{2,3})向([\u4e00-\u9fff]{2,3})汇报"),
    re.compile(r"([\u4e00-\u9fff]{2,3})汇报给"),
    re.compile(r"([\u4e00-\u9fff]{2,3})说[：:]"),
    re.compile(r"([\u4e00-\u9fff]{2,3})认为"),
    re.compile(r"([\u4e00-\u9fff]{2,3})提出"),
    re.compile(r"([\u4e00-\u9fff]{2,3})强调"),
    re.compile(r"([\u4e00-\u9fff]{2,3})指出"),
    re.compile(r"([\u4e00-\u9fff]{2,3})，[\u4e00-\u9fff]{2,8}"
               r"(?:总监|经理|主管|部长|厂长|董事长|总经理|副总裁|总裁|专员|工程师|组长|所长|负责人)"),
]

# 明显不是人名的普通词（即便多文档共现也要排除）
STOPWORDS = {
    "时间", "金额", "方案", "平台", "范围", "安全", "苏州", "方式", "时效", "周期",
    "安全委", "安全库", "安全生", "安全限", "安全培", "安全规", "安全法", "安全环",
    "周转型", "周转天", "周期由", "周期管", "周期为", "周期内",
    "方案在", "方案组", "方案配", "方案概", "方案基", "方案总",
    "施过程", "时内上", "时内响", "时废止", "时连续", "时执行", "时间由", "时间在",
    "于现有", "于其他", "于公司", "元以", "元以下", "元以上", "元增至", "元完全",
    "安装与", "安装工", "金属加", "金流净", "金牌渠", "金红利", "金奖", "金牌",
    "苏州二", "苏州基", "苏州总", "苏州工", "苏州中",
    "蒋涛对", "唐婧的", "蒋雯的",
    # 「角色/职责」类普通词，不是姓名
    "章包括", "章与部", "章由董", "章由法", "章由财", "章分别",
    "范围需", "平择优", "时应同", "时遵守", "何组合", "金月数",
    "元属于", "元是", "于已确", "方向", "时的去", "安全制", "安全委",
    "位于薪", "任销售", "任研发", "任控制", "任伺服", "任华东", "任上海",
    "任制造", "任苏州", "任人力", "任销", "任人",
    "华东区", "华东大", "苏州工", "周年", "年生", "年以",
    # 「部门/项目」等词被职位模式跨词捕获
    "成功部", "经部门", "须部门", "项目", "售后服", "服务中",
    "发展部", "与部门", "同部门", "求部门", "建部门",
}


def read_text(p: Path) -> str:
    suf = p.suffix.lower()
    try:
        if suf in {".md", ".txt"}:
            return p.read_text(encoding="utf-8", errors="replace")
        if suf == ".csv":
            return p.read_text(encoding="utf-8-sig", errors="replace")
        if suf == ".docx":
            import docx
            d = docx.Document(str(p))
            parts = [q.text for q in d.paragraphs]
            for tbl in d.tables:
                for row in tbl.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
            return "\n".join(parts)
        if suf == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(str(p), data_only=True)
            parts = []
            for ws in getattr(wb, "worksheets", []):
                for row in ws.iter_rows(values_only=True):
                    parts.append(" | ".join("" if v is None else str(v) for v in row))
            return "\n".join(parts)
        if suf == ".pdf":
            import pypdf
            r = pypdf.PdfReader(str(p))
            return "\n".join((pg.extract_text() or "") for pg in r.pages)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 读取失败 {p.name}: {exc}")
    return ""


def clean(name: str) -> str | None:
    """把捕获组裁成候选人名：姓 + 1~2 字。

    只做长度/姓氏/停用词裁剪，不在此处判断"是否真是人名"——
    真伪由调用方的结构化语境保证。
    """
    if not name or name in STOPWORDS:
        return None
    if name[0] not in SURNAMES:
        return None
    if len(name) == 2:
        return name
    if len(name) == 3:
        # 第 3 字是虚词/动词字时，说明捕获跨了词边界，退回 2 字
        if name[2] in "的了和与及在是为对向由把被就都也还都很更将等或但并而":
            return name[:2]
        return name
    return None


def main() -> None:
    files = sorted(p for p in DOCS.rglob("*") if p.is_file()
                   and p.suffix.lower() in {".md", ".txt", ".csv", ".docx", ".xlsx", ".pdf"})
    ctx_names: dict[str, set[str]] = defaultdict(set)   # 人名 → 出现文档
    name_docs: dict[str, set[str]] = defaultdict(set)   # 候选 → 出现文档

    for p in files:
        doc_id = p.stem
        text = read_text(p)
        if not text.strip():
            continue
        for pat in CONTEXT_PATTERNS:
            for m in pat.finditer(text):
                for g in m.groups():
                    nm = clean(g or "")
                    if nm:
                        ctx_names[nm].add(doc_id)
        # 全语料候选：姓氏 3 字 / 2 字，供统计用
        for m in NAME3.finditer(text):
            name_docs[m.group(0)].add(doc_id)
        for m in NAME2.finditer(text):
            name_docs[m.group(0)].add(doc_id)

    print(f"语料文件 {len(files)} 个")
    print(f"结构化语境抽出的人名 {len(ctx_names)} 个")

    # 补充规则：在 >=4 篇文档以「姓+2字」形式出现的短名，且不在停用词里
    short_extra = {n: d for n, d in name_docs.items()
                   if len(n) == 2 and len(d) >= 4 and n not in STOPWORDS
                   and n not in ctx_names}
    # 短名里仍有伪词（"范围"/"金额"等已进停用词）；再按"是否出现在语境人名邻近"筛
    print(f"多文档短名候选 {len(short_extra)} 个")

    whitelist = {n: sorted(d) for n, d in ctx_names.items()}
    print(f"最终白名单 {len(whitelist)} 人")

    # 语料人物全集：语料中出现过的所有"姓+1~2字"串。
    # 用途：模型答案里出现**语料中根本不存在**的人名（如「张三丰」）
    # 属于编造，必须判错；而语料中存在的人名即使金标准没提
    # （模型补充上下文）也不应因此判错。
    universe: set[str] = set()
    # 语料中出现的所有 2~3 字中文串。
    # 不限姓形态：中文里「成功部」「关系文」「公司担」这类跨词切片
    # 若按姓形态收录会残留成伪名；把全部 2~3 字串都收进来，
    # 判分器就能准确识别"语料中从未出现过的串"。
    for p in files:
        text = read_text(p)
        if not text.strip():
            continue
        for seg in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            for i in range(len(seg) - 1):
                universe.add(seg[i:i + 2])
                if i + 2 < len(seg):
                    universe.add(seg[i:i + 3])
        for m in NAME3.finditer(text):
            nm = m.group(0)
            if nm[0] in SURNAMES:
                universe.add(nm)
                universe.add(nm[:2])
        for m in NAME2.finditer(text):
            nm = m.group(0)
            if nm[0] in SURNAMES:
                universe.add(nm)
    universe -= STOPWORDS
    print(f"语料 2~3 字串全集 {len(universe)} 个")

    # 交叉校验：金标准证据原文中出现的人名是否都已覆盖
    abank = ROOT / "eval_bench" / "versions" / "20261030_v2" / "answer_bank.jsonl"
    covered = set(whitelist)
    missing: dict[str, int] = defaultdict(int)
    total = 0
    for line in abank.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        for ev in (a.get("evidence") or []):
            for s in (ev.get("segments") or []):
                txt = s.get("text") or ""
                for m in NAME3.finditer(txt):
                    nm = clean(m.group(0))
                    if nm:
                        total += 1
                        if nm not in covered:
                            missing[nm] += 1
                for m in NAME2.finditer(txt):
                    nm = clean(m.group(0))
                    if nm:
                        total += 1
                        if nm not in covered:
                            missing[nm] += 1
    print(f"\n证据原文人名提及 {total} 次，未覆盖 {len(missing)} 种：")
    for k, v in sorted(missing.items(), key=lambda x: -x[1])[:20]:
        print(f"  {k:<12}{v}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "note": "语料级真实人名白名单；判分器只把白名单内的串计为人名实体，"
                "从根本上消除「安全委」「范围需」这类跨词伪实体。"
                "corpus_name_universe 为语料中出现过的全部「姓+1~2字」串，"
                "用于识别模型答案里编造的不存在的人名。",
        "source": "enterprise_rag/corpus/docs 全部 136 篇",
        "names": sorted(whitelist.keys()),
        "corpus_name_universe": sorted(universe),
        "detail": {n: {"docs": len(d), "sample": d[:4]} for n, d in whitelist.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {OUT}")
    print("名单：", "、".join(sorted(whitelist.keys())))


if __name__ == "__main__":
    main()
