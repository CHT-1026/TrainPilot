"""Small, evidence-bound operators; no eval, no hidden model reasoning.

These operators handle explicit facts only. Unsupported syntax goes to synthesis;
missing operands or conflicting facts produce a refusal rather than a guessed total.
Every result includes source indices and a concise, reproducible calculation.
"""
import re
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation

from enterprise_rag.citations import chunk_body

CN = "零〇一二两三四五六七八九十百千万亿点"
NUMBER = rf"(?:\d[\d,]*(?:\.\d+)?|[{CN}]+)"
NAME = r"[\u4e00-\u9fff]{2,4}"


def number(text: str) -> Decimal:
    text = text.replace(",", "").replace("，", "").strip()
    try:
        return Decimal(text)
    except InvalidOperation:
        pass
    digits = {c: i for i, c in enumerate("零一二三四五六七八九")}
    digits.update({"〇": 0, "两": 2})
    if "点" in text:
        whole, fraction = text.split("点", 1)
        return number(whole) + Decimal("0." + "".join(str(digits[c]) for c in fraction))
    if all(c in digits for c in text):
        return Decimal("".join(str(digits[c]) for c in text))
    # Large units multiply the complete preceding group: 一万亿 = 10^12.
    for unit, multiplier in (("亿", 100000000), ("万", 10000)):
        if unit in text:
            left, right = text.split(unit, 1)
            return number(left or "一") * multiplier + number(right or "零")
    total = section = current = 0
    for char in text:
        if char in digits:
            current = digits[char]
        elif char in "十百千":
            section += (current or 1) * {"十": 10, "百": 100, "千": 1000}[char]
            current = 0
        else:
            raise ValueError("unsupported number")
    return Decimal(total + section + current)


def fmt(value) -> str:
    return format(Decimal(value).normalize(), "f")


def _result(text, indices, operation, facts=None, refused=False):
    indices = list(dict.fromkeys(indices))
    return {"answer": text, "indices": indices, "confidence": 1.0,
            "operation": operation, "facts": facts or [], "refused": refused}


def _refuse(reason, indices=()):
    return _result("现有企业资料不足以确定答案：" + reason, indices, "insufficient_evidence", refused=True)


def _body(chunk):
    return re.sub(r"(?m)^(?:所属部门|作者|更新时间|创建时间|生效日期|版本号|文档族)[:：].*$", "", chunk_body(chunk))


def _subjects(text):
    # Explicit subjects only; do not borrow a clock from another named company.
    return set(re.findall(r"([\u4e00-\u9fffA-Za-z0-9]{1,24}?(?:股份有限公司|有限公司|公司|项目))", text)) - {"本公司", "该公司", "本项目", "该项目"}


def _scoped(question, bodies, chunks):
    """Use explicit documentary subjects; unknown subjects never become wildcards.

    Titles may supply the subject of a paragraph saying '本公司'. A question
    mentioning one project must not aggregate another project's retrieved facts.
    Returning no scope is safer than silently interpreting all retrieved hits.
    """
    entries = [(i, text, _subjects(text + "\n" + str(chunks[i-1].get("title", "")))) for i, text in bodies]
    named = set().union(*(subjects for _, _, subjects in entries)) if entries else set()
    targets = {subject for subject in named if subject in question}
    if targets:
        selected = []
        for i, text, subjects in entries:
            if not subjects:
                selected.append((i, text))
                continue
            if not subjects & targets:
                continue
            active = _subjects(str(chunks[i-1].get("title", "")))
            pieces = []
            for sentence in re.split(r"(?<=[。；，\n])", text):
                explicit = _subjects(sentence)
                if explicit:
                    active = explicit
                if active & targets:
                    pieces.append(sentence)
            if pieces:
                selected.append((i, "".join(pieces)))
    elif len(named) <= 1:
        selected = bodies
    else:
        selected = []
    return selected, targets


def _record_subjects(i, text, chunks):
    return _subjects(text + "\n" + str(chunks[i-1].get("title", "")))


def _question_subjects(question):
    generic = {"这个公司", "这家公司", "那个公司", "那家公司", "哪家公司", "什么公司", "我们公司", "你们公司", "贵公司", "本公司", "该公司", "这个项目", "那个项目", "本项目", "该项目", "哪些项目", "所有项目", "海外项目", "相关项目"}
    return {subject for subject in _subjects(question) if not any(subject.endswith(word) for word in generic)}


def _department_count_intent(question):
    return bool(re.search(r"(?:多少|几)\s*个?(?:不同)?的?部门|部门(?:的)?(?:总数|数量|共有几个|有几个|共有多少个)|统计.{0,8}部门", question))


def _operator_intent(question):
    return bool(
        _department_count_intent(question)
        or re.search(r"(成立|创立|创办).{0,12}(年|多久)|多少年", question)
        or ("负责人" in question and "部门" in question)
        or re.search(r"(相隔多少天|自然日|结束日减开始日)", question)
        or (re.search(r"(合计|总计|相差|差额|多多少|少多少)", question) and re.search(r"(金额|合同|收入|元)", question))
    )


def _reporting_intent(question):
    return bool(re.search(
        r"(?:的上级|的直属上级|的上级的上级|的下属|直接下属|下属有(?:哪些|谁)|下属是谁|"
        r"向谁汇报|汇报给谁|汇报关系|什么关系|关系是什么|谁管理|归谁管理)",
        question,
    ))


def _reporting_relation(question, bodies, chunks):
    """Resolve explicit 上级/下属 edges into a bounded reporting chain.

    Facts are only those sentences that explicitly state a reporting edge.
    Missing or conflicting edges produce a refusal, never a guessed relation.
    """
    manager: dict[str, tuple[str, int]] = {}
    reports: dict[str, set[str]] = defaultdict(set)
    conflicts: set[str] = set()
    for index, text in bodies:
        for sentence in re.split(r"[。；\n]", text):
            for pattern, direction in (
                (rf"({NAME})(?:是|为)({NAME})的(?:直属)?上级", "down"),
                (rf"({NAME})的(?:直属)?上级(?:是|为)({NAME})", "up"),
                (rf"({NAME})(?:直接)?(?:向|对)({NAME})汇报", "up"),
                (rf"({NAME})(?:管理|带领|领导)({NAME})", "down"),
            ):
                for match in re.finditer(pattern, sentence):
                    first, second = match.group(1), match.group(2)
                    if first == second:
                        continue
                    person, boss = (first, second) if direction == "up" else (second, first)
                    if person in manager and manager[person][0] != boss:
                        conflicts.add(person)
                    manager.setdefault(person, (boss, index))
                    reports[boss].add(person)
    if not manager:
        return None
    if conflicts:
        return _refuse("汇报关系存在冲突记录，无法确定唯一上级。",
                       [manager[name][1] for name in conflicts])

    names = sorted(set(manager) | set(reports), key=len, reverse=True)
    mentioned = []
    for name in names:
        if name in question and name not in mentioned:
            mentioned.append(name)
    if not mentioned:
        return None

    def chain(person, limit=8):
        path = []
        current = person
        for _ in range(limit):
            if current not in manager:
                break
            boss, edge_index = manager[current]
            path.append((boss, edge_index, current))
            current = boss
        return path

    def descendants(person, limit=3):
        found = []
        frontier = [(person, 0)]
        seen = {person}
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= limit:
                continue
            for subordinate in sorted(reports.get(current, ())):
                if subordinate in seen:
                    continue
                seen.add(subordinate)
                found.append((subordinate, depth + 1))
                frontier.append((subordinate, depth + 1))
        return found

    if "下属" in question and len(mentioned) == 1 and "负责" in question:
        person = mentioned[0]
        people = descendants(person)
        if people:
            from app.capabilities.retrieval.tokenizer import tokenize

            stop = {"下属", "上级", "负责", "什么", "哪些", "哪个", "谁", "谁负责", "关系", "中谁"}
            terms = [term for term in tokenize(question)
                     if len(term) >= 2 and term not in stop and term != person]
            matches = []
            for subordinate, depth in people:
                for index, text in bodies:
                    for sentence in re.split(r"[。；\n]", text):
                        if subordinate in sentence and "负责" in sentence:
                            hits = sum(1 for term in terms if term in sentence)
                            if hits >= 2:
                                matches.append((hits, -depth, subordinate, sentence.strip(), index))
            if matches:
                matches.sort(reverse=True)
                hits, negative_depth, subordinate, sentence, index = matches[0]
                return _result(
                    f"{person}的第{-negative_depth}级下属中，{subordinate}负责相关业务："
                    f"{sentence} [{index}]。",
                    [index], "reporting_descendant_responsibility",
                )

    if "下属" in question and len(mentioned) == 1:
        person = mentioned[0]
        direct = sorted(reports.get(person, set()))
        if not direct:
            return _refuse(f"资料中没有{person}的下属记录。")
        segments = []
        citations = []
        for subordinate in direct:
            if subordinate not in manager:
                continue
            segments.append(f"{subordinate} [{manager[subordinate][1]}]")
            citations.append(manager[subordinate][1])
        if not segments:
            return _refuse(f"资料中没有{person}的下属记录。")
        return _result(
            f"{person}的直接下属有{len(segments)}人：{'、'.join(segments)}。",
            citations, "reporting_subordinates",
        )

    if "上级" in question and len(mentioned) == 1:
        person = mentioned[0]
        hops = min(question.count("上级"), 6)
        path = chain(person, hops)
        if not path:
            return _refuse(f"资料中没有{person}的上级记录。")
        if len(path) < hops:
            return _refuse(f"资料只记录到{person}的第{len(path)}级上级，无法确定更高层级。",
                           [item[1] for item in path])
        segments = []
        citations = []
        for boss, edge_index, subordinate in path[:hops]:
            segments.append(f"{subordinate}的上级是{boss} [{edge_index}]")
            citations.append(edge_index)
        final = path[hops - 1][0]
        if hops == 1:
            return _result(f"{person}的上级是{final} [{citations[0]}]。", citations,
                           "reporting_1_hop")
        return _result(
            "；".join(segments) + f"。因此{person}的上级" + ("的上级" * (hops - 1)) + f"是{final}。",
            citations, f"reporting_{hops}_hop",
        )

    if len(mentioned) == 2 and ("关系" in question or "汇报" in question):
        first, second = mentioned[0], mentioned[1]
        path_first = chain(first)
        path_second = chain(second)
        distance_first = next(
            (position + 1 for position, (boss, _, _) in enumerate(path_first) if boss == second), None
        )
        distance_second = next(
            (position + 1 for position, (boss, _, _) in enumerate(path_second) if boss == first), None
        )
        if distance_first:
            citations = [item[1] for item in path_first[:distance_first]]
            if distance_first == 1:
                text = f"{second}是{first}的直属上级"
            elif distance_first == 2:
                text = f"{second}是{first}的上级的上级（中间隔一级）"
            else:
                text = f"{second}是{first}的向上第{distance_first}级上级"
            evidence = "；".join(
                f"{subordinate}的上级是{boss} [{edge_index}]"
                for boss, edge_index, subordinate in path_first[:distance_first]
            )
            return _result(f"{text}。依据：{evidence}。", citations, "reporting_relation")
        if distance_second:
            citations = [item[1] for item in path_second[:distance_second]]
            if distance_second == 1:
                text = f"{first}是{second}的直属上级"
            elif distance_second == 2:
                text = f"{first}是{second}的上级的上级（中间隔一级）"
            else:
                text = f"{first}是{second}的向上第{distance_second}级上级"
            evidence = "；".join(
                f"{subordinate}的上级是{boss} [{edge_index}]"
                for boss, edge_index, subordinate in path_second[:distance_second]
            )
            return _result(f"{text}。依据：{evidence}。", citations, "reporting_relation")

        first_managers = {
            boss: (position, edge_index)
            for position, (boss, edge_index, _) in enumerate(path_first)
        }
        second_managers = {
            boss: (position, edge_index)
            for position, (boss, edge_index, _) in enumerate(path_second)
        }
        common = set(first_managers) & set(second_managers)
        if common:
            nearest = min(common, key=lambda boss: first_managers[boss][0] + second_managers[boss][0])
            first_position, first_index = first_managers[nearest]
            second_position, second_index = second_managers[nearest]
            if first_position == 0 and second_position == 0:
                return _result(
                    f"{first}和{second}是平级同事，均向{nearest}汇报 [{first_index}][{second_index}]。",
                    [first_index, second_index], "reporting_siblings",
                )
            return _result(
                f"{first}和{second}同属{nearest}管辖，但不是直接上下级关系 "
                f"[{first_index}][{second_index}]。",
                [first_index, second_index], "reporting_common_manager",
            )
        return _refuse("资料没有给出两人的汇报关系。")
    return None



def _company_founded(text):
    """Only a company's explicit founding statement, never an institute's date."""
    patterns = [
        r"(?:公司|企业)(?:正式)?(?:成立|创立|创办)(?:于|时间[:：]|日期[:：])?\s*((?:19|20)\d{2})\s*年",
        r"(?:公司|企业)(?:于)?\s*((?:19|20)\d{2})\s*年(?:\s*\d{1,2}\s*月\s*\d{1,2}\s*日)?(?:正式)?(?:成立|创立|创办)",
        r"((?:19|20)\d{2})\s*年(?:\s*\d{1,2}\s*月\s*\d{1,2}\s*日)?[，,]?\s*(?:本公司|该公司|公司|企业)(?:在[\u4e00-\u9fffA-Za-z·]{1,30})?(?:正式)?(?:成立|创立|创办)",
    ]
    return [int(m.group(1)) for pattern in patterns for m in re.finditer(pattern, text)]


def infer(question: str, chunks: list[dict], complete: bool = True) -> dict | None:
    bodies = [(i, _body(chunk)) for i, chunk in enumerate(chunks, 1)]
    if not bodies:
        return None
    if _reporting_intent(question):
        result = _reporting_relation(question, bodies, chunks)
        if result is not None:
            return result
    if not _operator_intent(question):
        return None
    scoped, targets = _scoped(question, bodies, chunks)
    known = set().union(*(_record_subjects(i, text, chunks) for i, text in bodies))
    if _question_subjects(question) and known and not targets:
        return _refuse("当前证据没有对应问题所指的公司或项目，不能借用其他主体的事实推断。")

    # A list is a set, not a count of search hits or file count.
    if _department_count_intent(question):
        if not scoped:
            return None
        if any(re.search(r"(仅摘录|节选|部分部门|部分样例|不是完整|完整.{0,6}未提供|不完整)", text) for _, text in scoped):
            return _refuse("部门资料只覆盖局部，不能推出公司全部部门总数。")
        entities = {}
        for i, text in scoped:
            if targets and not _record_subjects(i, text, chunks) & targets:
                continue
            for match in re.finditer(r"(?:设有|部门(?:为|是)|部门名称[:：])\s*([^，。；\n]{2,60})", text):
                for entity in re.split(r"[、]|(?<=[部所室])(?:和|与|及)", match.group(1)):
                    entity = entity.strip()
                    if entity.endswith(("部", "中心", "办公室", "研究所")):
                        entities.setdefault(entity, i)
        if entities:
            if not complete:
                return _refuse("相关清单超过本次检索/上下文预算，无法确认总数。")
            text = "；".join(f"{entity} [{i}]" for entity, i in entities.items())
            return _result(f"在本次可见组织清单范围内，去重后共 {len(entities)} 个部门：{text}。",
                           list(entities.values()), "count_unique", list(entities))

    # Relative foundation year requires a documentary clock for the SAME subject.
    if re.search(r"(成立|创立|创办).{0,12}(年|多久)|多少年", question):
        offsets = []
        clocks = []
        founded = []
        if not scoped:
            return None
        for i, text in scoped:
            subjects = _record_subjects(i, text, chunks)
            offsets.extend((i, number(m.group(1)), subjects) for m in re.finditer(
                rf"({NUMBER})\s*年前(?:.{{0,8}})(?:创立|成立|创办)", text))
            clocks.extend((i, int(m.group(1)), subjects) for m in re.finditer(
                r"(?:今年|当前年份|本年)(?:是|为|[:：])?\s*((?:19|20)\d{2})\s*年", text))
            founded.extend((i, year) for year in _company_founded(text))
        if offsets:
            years = set()
            indices = []
            for oi, offset, subjects in offsets:
                compatible = [(ci, year) for ci, year, other in clocks if (subjects & other) or (not subjects and not other)]
                if not compatible:
                    return _refuse("“年前”缺少同一主体的明确年份锚点，不能默认使用系统年份。")
                for ci, year in compatible:
                    years.add(Decimal(year) - offset)
                    indices.extend([oi, ci])
            if len(years) != 1:
                return _refuse("同一主体的年份锚点或年限存在冲突，请先核实版本。")
            year = years.pop()
            if founded and any(Decimal(value) != year for _, value in founded):
                return _refuse("直接记载的成立年份与相对年限推算存在冲突。")
            first_offset = offsets[0]
            first_clock = next(year_value for _, year_value, subjects in clocks if (subjects & first_offset[2]) or (not subjects and not first_offset[2]))
            formula = f"{first_clock} − {fmt(first_offset[1])} = {fmt(year)}"
            refs = "".join(f"[{i}]" for i in dict.fromkeys(indices))
            return _result(f"按资料中的年份锚点计算，公司成立于 {fmt(year)} 年；{formula}。{refs}", indices, "year_subtract")
        target_years = re.findall(r"((?:19|20)\d{2})\s*年", question)
        if founded and target_years and re.search(r"(经过|相隔|多少年|年限|多久)", question):
            base = {year for _, year in founded}
            if len(base) > 1:
                return _refuse("成立年份存在冲突。")
            start = base.pop()
            end = int(target_years[-1])
            indices = [i for i, _ in founded]
            return _result(f"公司成立于 {start} 年；按年份差计算，{end} − {start} = {end-start} 年（未按周年日计满年）。"
                           + "".join(f"[{i}]" for i in indices), indices, "year_difference")

    # Join a person named in the project record to an explicit department record.
    if "负责人" in question and "部门" in question:
        leaders = []
        if not scoped:
            return None
        for i, text in scoped:
            for sentence in re.split(r"[。；\n]", text):
                subjects = _subjects(sentence)
                if targets and subjects and not subjects & targets:
                    continue
                leaders.extend((i, m.group(1)) for m in re.finditer(r"负责人(?:是|为|[:：])\s*([\u4e00-\u9fff]{2,4})(?=[。；，\s]|$)", sentence))
        if leaders:
            joined = []
            for li, leader in leaders:
                for di, text in bodies:
                    m = re.search(re.escape(leader) + r"(?:的)?(?:所属部门|属于|隶属)(?:为|是|[:：])?\s*([^，。；\n]{2,30})", text)
                    if m:
                        joined.append((li, di, leader, m.group(1)))
            if not joined:
                return _refuse("已找到项目负责人，但缺少其所属部门记录。")
            departments = {j[3] for j in joined}
            if len(departments) != 1:
                return _refuse("负责人的部门归属存在冲突。")
            li, di, leader, department = joined[0]
            if "主管" in question:
                managers = []
                for mi, text in bodies:
                    pattern = re.escape(department) + r"(?:的)?(?:部门)?(?:主管|负责人)(?:是|为|[:：])\s*([\u4e00-\u9fff]{2,4})(?=[。；，\s]|$)"
                    managers.extend((mi, m.group(1)) for m in re.finditer(pattern, text))
                if not managers:
                    return _refuse("已找到负责人的所属部门，但缺少该部门主管的明确记录。")
                if len({name for _, name in managers}) != 1:
                    return _refuse("所属部门的主管记录存在冲突。")
                mi, manager = managers[0]
                return _result(f"项目负责人为{leader} [{li}]；{leader}属于{department} [{di}]；{department}主管为{manager} [{mi}]。因此项目负责人所属部门的主管是{manager} [{li}][{di}][{mi}]。",
                               [li, di, mi], "relational_join_three_hop")
            return _result(f"负责人是{leader} [{li}]；{leader}所属部门为{department} [{di}]，因此项目负责人的所属部门为{department} [{li}][{di}]。",
                           [li, di], "relational_join")

    # Calendar subtraction has explicit inclusive/exclusive semantics.
    if re.search(r"(相隔多少天|自然日|结束日减开始日)", question):
        starts, ends = [], []
        if not scoped:
            return None
        for i, text in scoped:
            for m in re.finditer(r"(开始|结束)日期[:：为是]?\s*(\d{4})[年-](\d{1,2})[月-](\d{1,2})日?", text):
                try:
                    value = date(*map(int, m.groups()[1:]))
                except ValueError:
                    return _refuse("资料包含无效日历日期。")
                (starts if m.group(1) == "开始" else ends).append((i, value))
        if starts or ends:
            if not starts or not ends:
                return _refuse("缺少开始日期或结束日期。")
            if len({v for _, v in starts}) > 1 or len({v for _, v in ends}) > 1:
                return _refuse("开始日期或结束日期存在冲突。")
            si, start = starts[0]
            ei, end = ends[0]
            if end < start:
                return _refuse("结束日期早于开始日期。")
            inclusive = bool(re.search(r"(都计入|均计入|首尾|含开始日|包含开始日)", question)) and "不包含开始日" not in question
            days = (end-start).days + int(inclusive)
            return _result(f"{end.isoformat()} − {start.isoformat()}"
                           + (" + 1" if inclusive else "") + f" = {days} 天"
                           + ("（开始日、结束日均计入）" if inclusive else "（不包含开始日）") + f"。[{si}][{ei}]",
                           [si, ei], "date_difference")

    if re.search(r"(合计|总计|相差|差额|多多少|少多少)", question) and re.search(r"(金额|合同|收入|元)", question):
        if not scoped:
            return None
        facts = {}
        qyears = set(re.findall(r"(?:19|20)\d{2}", question))
        requested_contracts = list(dict.fromkeys(re.findall(r"[甲乙丙丁A-Za-z0-9一二三四]+合同", question)))
        for i, text in scoped:
            if targets and not _record_subjects(i, text, chunks) & targets:
                continue
            for sentence in re.split(r"[。；\n]", text):
                if not re.search(r"金额|营业收入", sentence):
                    continue
                if "全年" in question and re.search(r"上半年|下半年|季度|单季|前三季度", sentence):
                    continue
                years = re.findall(r"(?:19|20)\d{2}", sentence)
                if qyears and years and not qyears.intersection(years):
                    continue
                for m in re.finditer(rf"(?P<label>[^。；：:\n]{{1,80}}?(?:金额|营业收入))\s*[:：为是]?\s*(?P<value>{NUMBER})\s*(?P<unit>亿元|万元|元)", sentence):
                    label = re.sub(r"\s+", "", m.group("label"))
                    if requested_contracts and not any(contract in label for contract in requested_contracts):
                        continue
                    if "预算" not in question and "预算" in label:
                        continue
                    if "预付款" not in question and "预付款" in label:
                        continue
                    if qyears and "营业收入" in label:
                        ys = re.findall(r"(?:19|20)\d{2}", label)
                        label = (ys[-1] + "营业收入") if ys else label
                    value = number(m.group("value")) * {"亿元": Decimal(100000000), "万元": Decimal(10000), "元": Decimal(1)}[m.group("unit")]
                    previous = facts.get(label)
                    if previous and previous[0] != value:
                        return _refuse("相同金额指标存在冲突。")
                    facts.setdefault(label, (value, i))
        if facts:
            difference = bool(re.search(r"(相差|差额|多多少|少多少)", question))
            explicit_pair = len(requested_contracts) == 2 or len(qyears) == 2
            if not complete and not (difference and explicit_pair and len(facts) == 2):
                return _refuse("金额证据超出检索预算，不能确认合计或差额。")
            if len(facts) < 2:
                return _refuse("运算需要至少两个独立金额，当前证据不足。")
            if requested_contracts and any(not any(contract in label for label in facts) for contract in requested_contracts):
                return _refuse("问题指定的合同金额未全部提供，不能用其他金额替代缺失运算数。")
            values = list(facts.values())
            if difference and len(values) != 2:
                return _refuse("差额的两个比较对象不唯一。")
            directional = difference and "比" in question and bool(re.search(r"(多多少|少多少)", question))
            if directional:
                if len(requested_contracts) == 2:
                    ordered_labels = [next(label for label in facts if contract in label) for contract in requested_contracts]
                elif len(qyears) == 2:
                    ordered_years = list(dict.fromkeys(re.findall(r"(?:19|20)\d{2}", question)))
                    ordered_labels = [next((label for label in facts if year in label), "") for year in ordered_years]
                    if not all(ordered_labels):
                        return _refuse("无法将比较主语与两个金额明确对应。")
                else:
                    return None
                values = [facts[label] for label in ordered_labels]
                if "少多少" in question:
                    values.reverse()
                total = values[0][0] - values[1][0]
            else:
                total = abs(values[0][0] - values[1][0]) if difference else sum(v for v, _ in values)
            divisor, unit = (Decimal(100000000), "亿元") if "多少亿元" in question else ((Decimal(10000), "万元") if "多少万元" in question else (Decimal(1), "元"))
            amounts = sorted([v/divisor for v, _ in values], reverse=True) if difference and not directional else [v/divisor for v, _ in values]
            indices = [i for _, i in values]
            formula = (" − " if difference else " + ").join(fmt(v) for v in amounts)
            return _result(f"统一单位后，{formula} = {fmt(total/divisor)} {unit}。" + "".join(f"[{i}]" for i in dict.fromkeys(indices)),
                           indices, "subtract" if difference else "sum", [{"label": k, "yuan": fmt(v), "citation": i} for k, (v, i) in facts.items()])
    return None
