"""制度问答节点：检索 -> 流式回答 -> 引用校验。"""
import logging

from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer

from app.capabilities.retrieval.answer import (
    REFUSAL_TEXT,
    build_prompt,
    format_with_sources,
    verify_answer,
)
from app.capabilities.retrieval.search import retrieve, to_citations
from app.core.config import get_settings
from app.core.metrics import inc
from app.workflows.chat_nodes.common import actor_from_state
from app.workflows.chat_nodes.enterprise_scope import looks_enterprise, looks_oa_policy
from app.workflows.chat_state import ChatState

logger = logging.getLogger(__name__)


def create_policy_nodes(llm: ChatOpenAI):
    nodes = {}

    def retrieve_node(state: ChatState):
        query = state.get("user_message", "")
        settings = get_settings()
        actor = actor_from_state(state)

        # 企业知识问题优先使用 lantu 语料（混合检索 + 版本消解 + 权限过滤）。
        if looks_enterprise(query) and not looks_oa_policy(query):
            try:
                from enterprise_rag import config as rag_config
                from enterprise_rag.retrieval import retrieve as enterprise_retrieve

                result = enterprise_retrieve(query, user=actor, top_k=settings.kb_top_k)
                enterprise_chunks = [
                    chunk
                    for chunk in result["chunks"]
                    if chunk.get("score", 0.0) >= rag_config.MIN_SCORE
                ]
                if enterprise_chunks:
                    inc("retrieval_hits_total")
                    logger.info(
                        "企业知识检索命中: %s 块，top=%.2f，query=%s",
                        len(enterprise_chunks),
                        enterprise_chunks[0].get("score", 0.0),
                        query[:50],
                    )
                    return {
                        "retrieved_chunks": enterprise_chunks,
                        "citations": [],
                        "task_result": "",
                        "grounded": False,
                    }
            except Exception as exc:
                logger.warning("企业知识检索失败，回退制度库: %s", exc)

        candidates = retrieve(query, actor, top_k=settings.kb_top_k)
        top_score = candidates[0].get("score", 0.0) if candidates else 0.0
        chunks = [chunk for chunk in candidates if chunk.get("score", 0.0) >= settings.kb_min_score]
        if not chunks:
            inc("retrieval_empty_total")
            logger.info("制度问答检索低于阈值: top=%.2f query=%s", top_score, query[:50])
            return {
                "retrieved_chunks": [],
                "citations": [],
                "task_result": REFUSAL_TEXT,
                "grounded": True,
            }
        inc("retrieval_hits_total")
        return {"retrieved_chunks": chunks, "task_result": ""}

    nodes["retrieve"] = retrieve_node

    async def answer_node(state: ChatState):
        chunks = state.get("retrieved_chunks") or []
        if not chunks:
            return {
                "task_result": REFUSAL_TEXT,
                "citations": [],
                "response": REFUSAL_TEXT,
                "grounded": True,
            }

        prompt = build_prompt(state.get("user_message", ""), chunks)
        writer = None
        try:
            writer = get_stream_writer()
        except Exception:
            writer = None

        parts: list[str] = []
        try:
            if writer is not None:
                async for delta in llm.astream(prompt):
                    text = delta.content if hasattr(delta, "content") else str(delta)
                    if text:
                        parts.append(text)
                        writer({"type": "token", "delta": text})
                answer = "".join(parts)
            else:
                response = await llm.ainvoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("制度问答生成失败: %s", exc, exc_info=True)
            try:
                response = await llm.ainvoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
            except Exception:
                answer = REFUSAL_TEXT

        final_answer, cited_chunks, grounded = verify_answer(answer, chunks)
        citations = to_citations(cited_chunks)
        text = format_with_sources(final_answer, cited_chunks)
        inc("policy_answers_total")
        if grounded:
            inc("policy_answers_grounded_total")
        logger.info(
            "制度问答完成: 引用 %s 条，grounded=%s",
            len(citations),
            grounded,
        )
        return {
            "task_result": text,
            "citations": citations,
            "response": text,
            "grounded": grounded,
        }

    nodes["answer"] = answer_node
    return nodes
