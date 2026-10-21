"""混合检索：BM25 + 向量 RRF 融合 → 权限/版本过滤 → 特征重排 → 文档去重。"""
import logging
import re
from collections import Counter, defaultdict

from app.domain.user.models import UserContext
from enterprise_rag import config
from enterprise_rag.graph.retrieve import graph_candidates
from enterprise_rag.query import query_terms, rewrite_query, semantic_query_variants, wants_table, reasoning_plan
from enterprise_rag.store import RagIndex, load_index, query_vector

logger = logging.getLogger(__name__)

SYSTEM_USER = UserContext(user_id="rag-eval", name="评测用户", role="admin", department="管理部")


def _allowed_rag(chunk: dict, user: UserContext) -> bool:
    """企业语料默认公司全员可见；department 仅作元数据，不参与跨部门拦截。"""
    if user.is_admin:
        return True
    visibility = chunk.get("visibility") or "all"
    if visibility == "all":
        return True
    if visibility == "manager":
        return user.is_manager
    if visibility == "hr":
        return user.is_hr
    if visibility == "finance":
        return user.role == "finance"
    if visibility == "admin":
        return False
    return False


def _family_filter(chunks: list[dict], index: RagIndex, as_of: str | None) -> list[dict]:
    allowed = index.allowed_doc_ids(as_of)
    if allowed is None:
        return chunks
    return [
        chunk
        for chunk in chunks
        if chunk.get("doc_family") not in index.versioned_families or chunk["doc_id"] in allowed
    ]


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


def _ensure_version_pair(selected: list[dict], reranked: list[dict], index: RagIndex) -> tuple[list[dict], str]:
    """Comparison questions need both sides of the same document family.

    Pick the versioned family that dominates the top ranked evidence, then make
    sure every document of that family contributes at least one chunk.
    """
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
        if chunk.get("doc_family") == family:
            best_by_doc.setdefault(chunk["doc_id"], chunk)
    if len(best_by_doc) < 2:
        return selected, family
    selected = list(selected)
    for doc_id, chunk in best_by_doc.items():
        if any(item["doc_id"] == doc_id for item in selected):
            continue
        replace_at = next(
            (
                position
                for position in range(len(selected) - 1, -1, -1)
                if selected[position].get("doc_family") != family
            ),
            len(selected) - 1,
        )
        if replace_at >= 0:
            selected[replace_at] = chunk
        else:
            selected.append(chunk)
    return selected, family



def retrieve(
    question: str,
    user: UserContext | None = None,
    top_k: int | None = None,
    as_of: str | None = None,
    namespace: str | None = None,
    retrieval_mode: str = "hybrid",
) -> dict:
    user = user or SYSTEM_USER
    top_k = top_k or config.TOP_K
    retrieval_mode = (retrieval_mode or "hybrid").lower()
    plan = reasoning_plan(question)
    if plan["complex"]:
        # Comparison needs both versions of one family, not the whole corpus.
        top_k = max(top_k, 12 if plan["comparison"] or plan["reporting"] else config.REASONING_TOP_K)
    index = load_index(namespace=namespace)
    rewritten = rewrite_query(question)
    effective_as_of = as_of or (rewritten["as_of"] if plan["policy_as_of"] else None)
    semantic_queries = semantic_query_variants(question)
    queries = list(dict.fromkeys(plan["queries"] + semantic_queries))[:8]
    terms = query_terms(" ".join([rewritten["rewritten"]] + queries), rewritten["original"])

    # Filter authorization and versions before applying the candidate budget.
    # Otherwise inaccessible records can crowd all readable evidence out of top K.
    def eligible(hits):
        hits = [c for c in hits if _allowed_rag(c, user)]
        if not plan["comparison"]:
            hits = _family_filter(hits, index, effective_as_of)
        if effective_as_of and (as_of or plan["strict_effective_date"]):
            hits = [c for c in hits if not c.get("effective_date") or str(c["effective_date"]) <= effective_as_of]
        return hits[:config.CANDIDATE_K]

    bm25_hits, vector_hits, merged = [], [], {}
    vector_errors = []
    for probe in queries:
        text = rewrite_query(probe)["rewritten"]
        b = eligible(index.bm25_search(text, max(config.CANDIDATE_K, len(index.chunks))))
        try:
            v = eligible(index.vector_search(query_vector(text), max(config.CANDIDATE_K, len(index.chunks))))
        except Exception as exc:
            # A provider failure must not mix incompatible vector spaces.
            logger.warning("向量检索不可用，显式降级 BM25: %s", type(exc).__name__)
            vector_errors.append(type(exc).__name__)
            v = []
        bm25_hits.extend(b)
        vector_hits.extend(v)
        for hit in _rrf(b, v):
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
    if plan["bridge"]:
        for hit in list(merged.values())[:12]:
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
    graph_debug = {"entities": [], "paths": [], "candidates": 0, "enabled": False,
                   "mode": retrieval_mode}
    graph_intent = bool(
        plan["reporting"] or plan["comparison"]
        or re.search(r"负责|属于|配套|客户|项目|版本|上级|下属|汇报|关系|架构", question)
    )
    if retrieval_mode == "graph" or (retrieval_mode == "hybrid" and graph_intent):
        graph_hits, graph_debug = graph_candidates(question, index, hops=2)
        graph_debug["mode"] = retrieval_mode
        allowed_graph = []
        for hit in graph_hits:
            if not _allowed_rag(hit, user):
                continue
            if plan["comparison"] or _family_filter([hit], index, effective_as_of):
                allowed_graph.append(hit)
        allowed_graph = allowed_graph[:40 if plan["comparison"] else 12]
        if retrieval_mode == "graph":
            fused = allowed_graph or fused
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
            fused = list(merged.values())
    if not plan["comparison"]:
        fused = _family_filter(fused, index, effective_as_of)
    reranked = _rerank(fused, terms, index, effective_as_of, wants_table(question), question)
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
    selected = _diversify(reranked, top_k)
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
    if plan["comparison"]:
        selected, comparison_family = _ensure_version_pair(selected, reranked, index)
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
            "evidence_budget_complete": relevant_ids <= selected_ids,
            "omitted_relevant_chunks": len(relevant_ids - selected_ids),
            "vector_fallback": bool(vector_errors),
            "vector_errors": sorted(set(vector_errors)),
        },
    }
