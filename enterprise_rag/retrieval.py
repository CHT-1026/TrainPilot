"""混合检索：BM25 + 向量 RRF 融合 → 权限/版本过滤 → 特征重排 → 文档去重。"""
import logging
import re
from collections import Counter, defaultdict

from app.domain.user.models import UserContext
from enterprise_rag import config
from enterprise_rag.budget import RequestBudget, bind_budget, new_budget
from enterprise_rag.citations import chunk_body
from enterprise_rag.facts import evidence_periods, factual_clauses, named_subjects
from enterprise_rag.graph.retrieve import graph_candidates
from enterprise_rag.query import query_terms, rewrite_query, semantic_query_variants, wants_table, reasoning_plan, topic_matches
from enterprise_rag.release import active_release, verify_release
from enterprise_rag.scope import EvidenceScope
from enterprise_rag.store import RagIndex, load_index, query_vector

logger = logging.getLogger(__name__)

SYSTEM_USER = UserContext(user_id="rag-eval", name="评测用户", role="admin", department="管理部")


def _rrf(bm25_hits: list[dict], vector_hits: list[dict]) -> list[dict]:
    scores: dict[str, float] = defaultdict(float)
    payload: dict[str, dict] = {}
    ranks: dict[str, dict] = defaultdict(dict)
    for rank, chunk in enumerate(bm25_hits, 1):
        chunk_id = chunk["chunk_id"]
        scores[chunk_id] += config.BM25_WEIGHT / (config.RRF_K + rank)
        ranks[chunk_id]["bm25_rank"] = rank
        payload[chunk_id] = chunk
    for rank, chunk in enumerate(vector_hits, 1):
        chunk_id = chunk["chunk_id"]
        scores[chunk_id] += config.VECTOR_WEIGHT / (config.RRF_K + rank)
        ranks[chunk_id]["vector_rank"] = rank
        ranks[chunk_id]["vector_score"] = chunk.get("vector_score", 0.0)
        payload.setdefault(chunk_id, chunk)

    merged = []
    for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        chunk = dict(payload[chunk_id])
        chunk["rrf_score"] = round(score, 6)
        chunk.update(ranks[chunk_id])
        merged.append(chunk)
    return merged


def _rerank(
    chunks: list[dict],
    terms: list[str],
    index: RagIndex,
    as_of: str | None,
    table_query: bool,
    question: str,
) -> list[dict]:
    comparison = reasoning_plan(question)["comparison"]
    term_weights = {
        term: max(index.term_idf.get(term, 1.0), 0.8) * (1.25 if term.isascii() else 1.0)
        for term in terms
    }
    total_weight = sum(term_weights.values()) or 1.0
    entity_terms = [
        term
        for term in terms
        if any(char.isalpha() for char in term)
        and any(char.isdigit() for char in term)
        and not term.startswith("20")
    ]
    numeric_intent = any(
        hint in question
        for hint in ("多少", "几台", "金额", "比例", "标准", "参数", "时间", "多久", "人数", "价格", "收入", "利润")
    )
    question_years = set(re.findall(r"(20\d{2})", question))
    month_match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", question)
    quarter_match = re.search(r"(20\d{2})\s*年\s*第?\s*([一二三四1-4])\s*季度", question)
    exact_month = f"{month_match.group(1)}年{int(month_match.group(2))}月" if month_match else ""
    quarter_map = {"一": "一", "二": "二", "三": "三", "四": "四", "1": "一", "2": "二", "3": "三", "4": "四"}
    exact_quarter = (
        f"{quarter_match.group(1)}年第{quarter_map[quarter_match.group(2)]}季度"
        if quarter_match
        else ""
    )
    reranked = []
    for chunk in chunks:
        text = chunk.get("text", "")
        text_tokens = set(chunk.get("_tokens") or [])
        if not text_tokens:
            from app.capabilities.retrieval.tokenizer import tokenize

            text_tokens = set(tokenize(text))
            chunk["_tokens"] = list(text_tokens)
        title_tokens = set(chunk.get("_title_tokens") or [])
        if not title_tokens:
            from app.capabilities.retrieval.tokenizer import tokenize

            title_tokens = set(tokenize(f"{chunk.get('title', '')} {chunk.get('section', '')}"))
            chunk["_title_tokens"] = list(title_tokens)

        matched_weight = sum(weight for term, weight in term_weights.items() if term in text_tokens)
        coverage = matched_weight / total_weight
        title_hits = sum(1 for term in terms if term in title_tokens)
        title_score = min(title_hits / max(len(terms), 1) * 2, 1.0)
        vector_score = float(chunk.get("vector_score", 0.0))
        phrase_hits = 0
        # 原始问题中的关键词直接出现在文本里，给予短语命中奖励。
        for term in terms[:8]:
            if term in text:
                phrase_hits += 1
        phrase_score = min(phrase_hits / 4, 1.0)
        table_bonus = 0.05 if table_query and chunk.get("chunk_type") == "table" else 0.0
        entity_bonus = 0.0
        if entity_terms:
            hits = sum(1 for term in entity_terms if term in text_tokens)
            entity_bonus = min(hits * 0.10, 0.20)
        numeric_bonus = 0.0
        if numeric_intent and re.search(r"\d[\d,\.]*\s*(台|元|万元|亿元|%|小时|分钟|天|人|kg|mm)", text):
            numeric_bonus = 0.08
        graph_bonus = min(float(chunk.get("graph_score", 0.0)) * 0.35, 0.22)
        model_scope_penalty = 0.0
        for model in re.findall(r"\b(?:LT|Lantu)[-A-Za-z0-9]*[A-Za-z0-9]+(?:[-A-Za-z0-9]+)*\b", question):
            suffix = r"\s*(?:Pro|Plus|Max|II|2\.0)"
            if re.search(re.escape(model) + suffix, text, re.I) and not re.search(
                re.escape(model) + suffix, question, re.I
            ):
                model_scope_penalty = -0.30
                break
        year_bonus = 0.0
        if question_years:
            text_years = set(re.findall(r"(20\d{2})", text))
            if text_years:
                year_bonus = 0.06 if text_years & question_years else -0.06
        date_bonus = 0.0
        if exact_month and exact_month in text.replace(" ", ""):
            date_bonus = 0.16
        elif exact_quarter and exact_quarter in text.replace(" ", ""):
            date_bonus = 0.16
        latest_bonus = 0.0
        family = chunk.get("doc_family")
        if not comparison and family and family in index.versioned_families and not as_of:
            latest_doc = max(
                index.family_docs[family],
                key=lambda item: (item["doc_updated_at"], item["version"]),
            )
            if latest_doc["doc_id"] == chunk["doc_id"]:
                latest_bonus = 0.03

        score = (
            0.40 * coverage
            + 0.30 * vector_score
            + 0.15 * title_score
            + 0.10 * phrase_score
            + table_bonus
            + entity_bonus
            + numeric_bonus
            + graph_bonus
            + model_scope_penalty
            + year_bonus
            + date_bonus
            + latest_bonus
        )
        chunk["score"] = round(score, 6)
        chunk["features"] = {
            "coverage": round(coverage, 4),
            "vector": round(vector_score, 4),
            "title": round(title_score, 4),
            "phrase": round(phrase_score, 4),
            "table_bonus": table_bonus,
            "entity_bonus": round(entity_bonus, 4),
            "numeric_bonus": numeric_bonus,
            "graph_bonus": round(graph_bonus, 4),
            "model_scope_penalty": model_scope_penalty,
            "year_bonus": year_bonus,
            "date_bonus": date_bonus,
            "latest_bonus": latest_bonus,
        }
        reranked.append(chunk)
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked


def _diversify(chunks: list[dict], top_k: int) -> list[dict]:
    selected = []
    per_doc: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        if per_doc[doc_id] >= config.MAX_CHUNKS_PER_DOC:
            continue
        per_doc[doc_id] += 1
        selected.append(chunk)
        if len(selected) >= top_k:
            break
    return selected


def _topic_supported(topic: dict, chunk: dict, question: str, subjects: list[str]) -> bool:
    body = chunk_body(chunk)
    identity = body + " " + str(chunk.get("title", ""))
    if any(subject not in identity for subject in subjects):
        return False
    years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", question))
    planning = bool(re.search(r"目标|计划|预算|预计|预测", question))
    for part in factual_clauses(body):
        if not planning and re.search(r"目标|计划|预算|预计|预测|将保持|力争", part):
            continue
        if years and not years.intersection(evidence_periods(part, chunk)):
            continue
        if topic_matches(topic["id"], part):
            return True
    return False


def _cover_topics(chunks: list[dict], top_k: int, plan: dict, question: str,
                  ranks: dict[str, dict[str, int]]) -> list[dict]:
    """Reserve one legal substantive hit per requested topic, then fill by rank.

    This does not search outside the already scoped candidates, and it does not
    claim the evidence proves an answer or a causal relationship.
    """
    selected, ids, content_keys = [], set(), set()
    per_doc: dict[str, int] = defaultdict(int)

    def add(chunk):
        body = re.sub(r"\s+", "", chunk_body(chunk))
        key = (body, tuple(sorted(evidence_periods(chunk_body(chunk), chunk))),
               tuple(sorted(named_subjects(chunk_body(chunk) + " " + str(chunk.get("title", ""))))),
               chunk.get("namespace", ""))
        if (len(selected) >= top_k or chunk["chunk_id"] in ids or key in content_keys
                or per_doc[chunk["doc_id"]] >= config.MAX_CHUNKS_PER_DOC):
            return False
        selected.append(chunk)
        ids.add(chunk["chunk_id"])
        content_keys.add(key)
        per_doc[chunk["doc_id"]] += 1
        return True

    for topic in plan["topics"]:
        if any(_topic_supported(topic, c, question, plan["topic_subjects"]) for c in selected):
            continue
        candidates = [c for c in chunks if _topic_supported(topic, c, question, plan["topic_subjects"])]
        candidates.sort(key=lambda c: (ranks.get(topic["id"], {}).get(c["chunk_id"], 10**9),
                                       -float(c.get("score", 0))))
        for candidate in candidates:
            if add(candidate):
                break
    for chunk in chunks:
        add(chunk)
    return sorted(selected, key=lambda c: float(c.get("score", 0)), reverse=True)


def _topic_coverage(selected: list[dict], plan: dict, question: str, used_queries: list[str]) -> dict:
    sources = {topic["id"]: [c["chunk_id"] for c in selected
                            if _topic_supported(topic, c, question, plan["topic_subjects"])]
               for topic in plan["topics"]}
    missing = [key for key, ids in sources.items() if not ids]
    return {"applicable":bool(sources), "requested":list(sources), "covered":[key for key, ids in sources.items() if ids],
            "missing":missing, "complete":not missing if sources else None, "source_chunk_ids":sources,
            "topic_queries_used":[topic["id"] for topic in plan["topics"] if topic["query"] in used_queries],
            "meaning":"每个所问主题至少有一块同主体/统计期的可见实质证据；不代表完整事实或因果证明"}


_VERSION_QUESTION = re.compile(
    r"20\d{2}.{0,12}20\d{2}|20\d{2}\s*版|旧版|新版|上一版|历史版本|哪一版|版本对比|版本差异|"
    r"新增|增加|减少|取消|升级|改版|变更|"
    r"20\d{2}\s*年.{0,6}(?:更新|发布|生效|改版)|更新于\s*20\d{2}"
)


def _ensure_version_pair(selected: list[dict], reranked: list[dict], index: RagIndex,
                         question: str = "", scope: EvidenceScope | None = None) -> tuple[list[dict], str]:
    """Comparison questions need both sides of the same document family.

    Only already-authorized candidates are used; the pair is completed inside the
    same retrieval call rather than re-reading the global index at generation time.
    """
    from enterprise_rag.evidence import version_pair

    family = ""
    pair = version_pair(reranked[:24], question) if question else None
    if pair:
        family = str(pair[0].get("doc_family") or "")
    if not family:
        counts = Counter(
            chunk.get("doc_family")
            for chunk in reranked[:12]
            if chunk.get("doc_family") in index.versioned_families
        )
        if not counts:
            return selected, ""
        family = counts.most_common(1)[0][0]
    docs = index.family_docs.get(family, [])
    if len(docs) < 2:
        return selected, ""
    best_by_doc: dict[str, dict] = {}
    for chunk in reranked:
        if chunk.get("doc_family") != family:
            continue
        current = best_by_doc.get(chunk["doc_id"])
        if current is None or _version_chunk_rank(chunk, family) > _version_chunk_rank(current, family):
            best_by_doc[chunk["doc_id"]] = chunk
    # 同一检索调用内按标题补查缺失版本；所有命中仍经过 EvidenceScope。
    if scope is not None:
        for doc in docs:
            doc_id = doc["doc_id"]
            if doc_id in best_by_doc:
                continue
            probe = f"{doc.get('title', '')} {doc.get('version', '')}".strip()
            hits = scope.filter(index.bm25_search(probe, max(config.CANDIDATE_K, len(index.chunks))),
                                index)
            candidates = [hit for hit in hits if hit.get("doc_id") == doc_id]
            if candidates:
                best_by_doc[doc_id] = max(
                    candidates, key=lambda item: _version_chunk_rank(item, family)
                )
    if len(best_by_doc) < 2:
        return selected, family
    present = {chunk["doc_id"] for chunk in selected if chunk.get("doc_family") == family}
    missing = {doc_id: chunk for doc_id, chunk in best_by_doc.items() if doc_id not in present}
    if not missing:
        return selected, family
    selected = list(selected)
    for chunk in missing.values():
        replace_at = next(
            (
                position
                for position in range(len(selected) - 1, -1, -1)
                if selected[position].get("doc_family") != family
            ),
            None,
        )
        if replace_at is None:
            selected.append(chunk)
        else:
            selected[replace_at] = chunk
    return selected, family


def _version_chunk_rank(chunk: dict, family: str) -> tuple[int, float]:
    """组织架构优先含一级部门清单的块，其余按检索分取最高块。"""
    has_list = 1 if family == "org_chart" and "个一级部门" in str(chunk.get("text") or "") else 0
    return has_list, float(chunk.get("score") or 0.0)



def _neighbor_expand(selected: list[dict], candidates: list[dict], index: RagIndex,
                     scope: EvidenceScope, top_k: int) -> tuple[list[dict], int]:
    """按文档 ord 扩展相邻块；只接受同一合法边界内的证据。"""
    if not config.NEIGHBOR_EXPANSION:
        return selected, 0
    expanded = list(selected)
    seen = {chunk["chunk_id"] for chunk in expanded}
    candidate_ids = {chunk["chunk_id"] for chunk in candidates}
    added = 0
    for chunk in selected:
        if len(expanded) >= top_k:
            break
        doc_chunks = index.by_doc_ord.get(chunk.get("doc_id"), [])
        position = chunk.get("ord", 0)
        for offset in range(1, config.NEIGHBOR_WINDOW + 1):
            for neighbor_ord in (position - offset, position + offset):
                neighbor = next((item for item in doc_chunks if item.get("ord") == neighbor_ord), None)
                if not neighbor or neighbor["chunk_id"] in seen:
                    continue
                if not scope.filter([neighbor], index):
                    continue
                if neighbor["chunk_id"] not in candidate_ids:
                    neighbor = dict(neighbor, score=min(chunk.get("score", 0.0) * 0.6, 0.45),
                                    features={"neighbor_of": chunk["chunk_id"]})
                if len(expanded) >= top_k:
                    return expanded, added
                expanded.append(neighbor)
                seen.add(neighbor["chunk_id"])
                added += 1
    return expanded[:top_k], added


def retrieve(
    question: str,
    user: UserContext | None = None,
    top_k: int | None = None,
    as_of: str | None = None,
    namespace: str | None = None,
    retrieval_mode: str = "hybrid",
    budget: RequestBudget | None = None,
    offline: bool = False,
) -> dict:
    budget = budget or new_budget(offline=offline)
    with bind_budget(budget):
        with budget.stage("retrieval"):
            result = _retrieve_impl(question, user, top_k, as_of, namespace, retrieval_mode, budget)
    result["debug"]["budget"] = budget.snapshot()
    return result


def _retrieve_impl(
    question: str,
    user: UserContext | None,
    top_k: int | None,
    as_of: str | None,
    namespace: str | None,
    retrieval_mode: str,
    budget: RequestBudget,
) -> dict:
    user = user or SYSTEM_USER
    top_k = top_k or config.TOP_K
    retrieval_mode = (retrieval_mode or "hybrid").lower()
    plan = reasoning_plan(question)
    if plan["complex"] and not plan.get("bounded_topic_only"):
        # Comparison needs both versions of one family, not the whole corpus.
        top_k = max(top_k, 12 if plan["comparison"] or plan["reporting"] else config.REASONING_TOP_K)
    index = load_index(namespace=namespace)
    rewritten = rewrite_query(question)
    effective_as_of = as_of or (rewritten["as_of"] if plan["policy_as_of"] else None)
    release = active_release(namespace)
    scope = EvidenceScope.from_request(
        user,
        namespace=namespace,
        as_of=effective_as_of,
        strict_effective_date=bool(as_of or plan["strict_effective_date"]),
        comparison=bool(plan["comparison"] or _VERSION_QUESTION.search(question)),
        snapshot=release.get("snapshot_id", ""),
        acl_version=(f"{release.get('acl_version', '')}:{index.access_revision}"
                     if getattr(index, "access_revision", "") else release.get("acl_version", "")),
    )
    semantic_queries = semantic_query_variants(question, allow_llm=budget.can_call_model())
    if config.HYDE_ENABLED and budget.can_call_model():
        from enterprise_rag.query import hyde_variants

        semantic_queries = list(dict.fromkeys(
            semantic_queries + hyde_variants(question, allow_llm=budget.can_call_model())
        ))
    queries = list(dict.fromkeys(plan["queries"] + semantic_queries))[:config.MAX_QUERY_VARIANTS]
    terms = query_terms(" ".join([rewritten["rewritten"]] + queries), rewritten["original"])
    # 同块数换版也必须停用：索引对象记录加载时的权威版本，不能只核对块数。
    # 未绑定版本的索引（如测试替身）无法与发布快照比对，不伪造不一致。
    bound_version = getattr(index, "index_version", "")
    stale_index = []
    if bound_version and release.get("index_version") and bound_version != release["index_version"]:
        stale_index.append({"code": "INDEX_VERSION_MISMATCH", "detail": "索引已变化，需重新发布"})
    if bound_version and release.get("index_chunks") is not None \
            and int(release["index_chunks"]) != len(index.chunks):
        stale_index.append({"code": "INDEX_CHUNK_COUNT_MISMATCH", "detail": "块数与发布快照不一致"})
    if stale_index:
        return {
            "query": rewritten, "as_of": effective_as_of, "terms": terms,
            "candidates": 0, "chunks": [], "scope": scope.to_dict(),
            "debug": {
                "bm25_hits": 0, "vector_hits": 0, "fused": 0, "reranked": 0,
                "top_score": 0.0, "comparison_mode": plan["comparison"],
                "evidence_budget_complete": False,
                "graph": {"release_issues": stale_index, "paths": []},
                "plan": {**plan, "bridge_queries": [], "semantic_queries": semantic_queries},
                "queries_used": queries,
                "queries_executed": [],
                "topic_coverage": _topic_coverage([], plan, question, []),
                "release_issues": stale_index,
            },
        }

    def eligible(hits):
        return scope.filter(hits, index)[:config.CANDIDATE_K]

    bm25_hits, vector_hits, merged = [], [], {}
    topic_by_query = {topic["query"]:topic["id"] for topic in plan["topics"]}
    topic_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    executed_queries = []
    vector_errors = []
    budget_stops = []
    graph_only = retrieval_mode == "graph"
    for probe in ([] if graph_only else queries):
        if budget.expired():
            budget.stop("deadline_exceeded")
            budget_stops.append("deadline_exceeded")
            break
        text = rewrite_query(probe)["rewritten"]
        executed_queries.append(probe)
        b = eligible(index.bm25_search(text, max(config.CANDIDATE_K, len(index.chunks))))
        if config.MULTI_REPRESENTATION and hasattr(index, "representation_search"):
            b = eligible(b + index.representation_search(text, config.CANDIDATE_K))
        v = []
        if budget.can_call_embedding():
            try:
                v = eligible(index.vector_search(query_vector(text), max(config.CANDIDATE_K, len(index.chunks))))
            except Exception as exc:
                # A provider failure must not mix incompatible vector spaces.
                logger.warning("向量检索不可用，显式降级 BM25: %s", type(exc).__name__)
                vector_errors.append(type(exc).__name__)
                v = []
        else:
            vector_errors.append("EmbeddingBudgetExceeded")
            budget_stops.append("embedding_budget_exhausted")
            budget.stop("embedding_budget_exhausted")
        bm25_hits.extend(b)
        vector_hits.extend(v)
        for rank, hit in enumerate(_rrf(b, v), 1):
            if probe in topic_by_query:
                topic_ranks[topic_by_query[probe]][hit["chunk_id"]] = rank
            current = merged.get(hit["chunk_id"])
            if current is None:
                hit["query_hits"] = 1
                merged[hit["chunk_id"]] = hit
            else:
                # 多查询 RRF：同一证据被多个语义变式命中时累计排名分。
                current["rrf_score"] = round(current.get("rrf_score", 0.0) + hit["rrf_score"], 6)
                current["query_hits"] = int(current.get("query_hits", 1)) + 1
    # A bounded second hop follows entities found in authorized first-hop evidence.
    bridge_queries = []
    if plan["bridge"] and plan.get("max_hops", 1) >= 2 and not budget.expired():
        for hit in list(merged.values())[:12]:
            if len(bridge_queries) >= 3:
                break
            for name in re.findall(r"负责人(?:是|为|[:：])\s*([\u4e00-\u9fff]{2,4})(?=[。；，\s]|$)", hit.get("text", "")):
                probe = f"{name} 所属部门 员工 部门归属"
                if probe in bridge_queries:
                    continue
                bridge_queries.append(probe)
                for hit2 in eligible(index.bm25_search(probe, max(config.CANDIDATE_K, len(index.chunks)))):
                    merged.setdefault(hit2["chunk_id"], dict(hit2, rrf_score=0.0))
                if len(bridge_queries) >= 3:
                    break
    fused = list(merged.values())
    graph_debug = {"entities": [], "paths": [], "path_evidence": [], "candidates": 0, "enabled": False,
                   "mode": retrieval_mode}
    graph_intent = bool(
        plan["reporting"] or plan["comparison"]
        or re.search(r"负责|属于|配套|客户|项目|版本|上级|下属|汇报|关系|架构", question)
    )
    if retrieval_mode == "graph" or (retrieval_mode == "hybrid" and graph_intent):
        graph_hits, graph_debug = graph_candidates(
            question, index, hops=min(plan.get("max_hops", config.MAX_HOPS), config.MAX_HOPS), scope=scope
        )
        graph_debug["mode"] = retrieval_mode
        release_issues = verify_release(
            release,
            graph_stats={"version": graph_debug.get("graph_version", "")},
        )
        if release_issues:
            graph_debug["release_issues"] = release_issues
            graph_hits = []
        allowed_graph = []
        for hit in graph_hits:
            if not scope.filter([hit], index):
                continue
            allowed_graph.append(hit)
        allowed_graph = allowed_graph[:40 if plan["comparison"] else 12]
        # 防御性重建：即使图支路返回了未过滤的轨迹，也只保留合法候选携带的路径。
        graph_debug["paths"] = list(dict.fromkeys(
            path for hit in allowed_graph for path in (hit.get("graph_paths") or [])
        ))[:12]
        graph_debug["path_evidence"] = [
            evidence for hit in allowed_graph
            for evidence in (hit.get("graph_path_evidence") or [])
        ][:12]
        if retrieval_mode == "graph":
            # 图模式必须真实只返回图证据，不再静默回退到向量候选。
            fused = allowed_graph
            graph_debug["graph_only"] = True
            graph_debug["vector_fallback"] = False
        else:
            for hit in allowed_graph:
                current = merged.get(hit["chunk_id"])
                if current is None:
                    merged[hit["chunk_id"]] = hit
                else:
                    current["graph_score"] = max(float(current.get("graph_score", 0)),
                                                 float(hit.get("graph_score", 0)))
                    if hit.get("graph_paths"):
                        current["graph_paths"] = hit["graph_paths"]
                    if hit.get("graph_path_evidence"):
                        current["graph_path_evidence"] = hit["graph_path_evidence"]
            fused = list(merged.values())
    fused = scope.filter(fused, index)
    reranked = _rerank(fused, terms, index, effective_as_of, wants_table(question), question)
    if config.CROSS_ENCODER_ENABLED:
        from enterprise_rag.rerank import lexical_pair_rerank

        reranked = lexical_pair_rerank(question, reranked)
    if plan["enumeration"]:
        # One chunk per document first, then fill spare budget with other sections.
        first, rest, seen = [], [], set()
        for chunk in reranked:
            if chunk["doc_id"] in seen:
                rest.append(chunk)
            else:
                first.append(chunk)
                seen.add(chunk["doc_id"])
        reranked = first + rest
    selected = (_cover_topics(reranked, top_k, plan, question, topic_ranks)
                if plan["topics"] else _diversify(reranked, top_k))
    if retrieval_mode == "graph" and plan["comparison"]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for chunk in reranked:
            if chunk.get("graph_document"):
                grouped[chunk["doc_id"]].append(chunk)
        if len(grouped) >= 2:
            selected = []
            for doc_id, chunks in grouped.items():
                chunks.sort(key=lambda item: (item.get("graph_priority", 99), item.get("ord", 0)))
                selected.extend(chunks[:4])
            selected = selected[:top_k]
            selected_ids_now = {chunk["chunk_id"] for chunk in selected}
            for chunk in reranked:
                if len(selected) >= top_k:
                    break
                if chunk["chunk_id"] not in selected_ids_now:
                    selected.append(chunk)
                    selected_ids_now.add(chunk["chunk_id"])
    comparison_family = ""
    if plan["comparison"] or _VERSION_QUESTION.search(question):
        selected, comparison_family = _ensure_version_pair(selected, reranked, index, question, scope)
    selected, neighbor_added = _neighbor_expand(selected, reranked, index, scope, top_k)
    relevant_ids = {c["chunk_id"] for c in reranked if c.get("score", 0) >= config.MIN_SCORE}
    selected_ids = {c["chunk_id"] for c in selected}

    for chunk in selected:
        chunk.pop("_tokens", None)
        chunk.pop("_title_tokens", None)
        chunk.pop("embedding", None)

    return {
        "query": rewritten,
        "as_of": effective_as_of,
        "terms": terms,
        "candidates": len(fused),
        "chunks": selected,
        "scope": scope.to_dict(),
        "debug": {
            "bm25_hits": len(bm25_hits),
            "vector_hits": len(vector_hits),
            "fused": len(fused),
            "reranked": len(reranked),
            "top_score": selected[0]["score"] if selected else 0.0,
            "versioned_families": sorted(index.versioned_families),
            "comparison_mode": plan["comparison"],
            "comparison_family": comparison_family,
            "retrieval_mode": retrieval_mode,
            "graph": graph_debug,
            "plan": {**plan, "bridge_queries": bridge_queries, "semantic_queries": semantic_queries},
            "queries_used": queries,
            "queries_executed": executed_queries,
            "topic_coverage": _topic_coverage(selected, plan, question, executed_queries),
            "evidence_budget_complete": relevant_ids <= selected_ids,
            "omitted_relevant_chunks": len(relevant_ids - selected_ids),
            "vector_fallback": bool(vector_errors),
            "vector_errors": sorted(set(vector_errors)),
            "budget_stops": sorted(set(budget_stops)),
            "neighbor_expanded": neighbor_added,
            "scope": scope.to_dict(),
            "feature_flags": {
                "multi_representation": config.MULTI_REPRESENTATION,
                "neighbor_expansion": config.NEIGHBOR_EXPANSION,
                "cross_encoder": config.CROSS_ENCODER_ENABLED,
                "hyde": config.HYDE_ENABLED,
                "context_compression": config.CONTEXT_COMPRESSION,
            },
        },
    }

