"""混合检索：BM25 + 向量 RRF 融合 → 权限/版本过滤 → 特征重排 → 文档去重。"""
import logging
import re
from collections import defaultdict

from app.domain.user.models import UserContext
from enterprise_rag import config
from enterprise_rag.query import query_terms, rewrite_query, wants_table
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
        if family and family in index.versioned_families and not as_of:
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


def retrieve(
    question: str,
    user: UserContext | None = None,
    top_k: int | None = None,
    as_of: str | None = None,
    namespace: str | None = None,
) -> dict:
    user = user or SYSTEM_USER
    top_k = top_k or config.TOP_K
    index = load_index(namespace=namespace)
    rewritten = rewrite_query(question)
    effective_as_of = as_of or rewritten["as_of"]
    terms = query_terms(rewritten["rewritten"], rewritten["original"])

    bm25_hits = index.bm25_search(rewritten["rewritten"], config.CANDIDATE_K)
    vector_hits = index.vector_search(query_vector(rewritten["rewritten"]), config.CANDIDATE_K)

    bm25_hits = [chunk for chunk in bm25_hits if _allowed_rag(chunk, user)]
    vector_hits = [chunk for chunk in vector_hits if _allowed_rag(chunk, user)]
    bm25_hits = _family_filter(bm25_hits, index, effective_as_of)
    vector_hits = _family_filter(vector_hits, index, effective_as_of)

    fused = _rrf(bm25_hits, vector_hits)
    fused = _family_filter(fused, index, effective_as_of)
    reranked = _rerank(fused, terms, index, effective_as_of, wants_table(question), question)
    selected = _diversify(reranked, top_k)

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
        },
    }
