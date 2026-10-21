"""Knowledge nodes: shared enterprise reasoning, current-user retrieval, verified output."""
import asyncio
import logging

from langgraph.config import get_stream_writer

from app.capabilities.retrieval.answer import REFUSAL_TEXT, build_prompt, format_with_sources, verify_answer
from app.capabilities.retrieval.search import retrieve, to_citations
from app.core.config import get_settings
from app.core.metrics import inc
from app.workflows.chat_nodes.common import actor_from_state
from app.workflows.chat_nodes.enterprise_scope import looks_enterprise, looks_oa_policy
from app.workflows.chat_state import ChatState
from enterprise_rag.query import contextualize

logger = logging.getLogger(__name__)


def create_policy_nodes(llm):
    def retrieve_node(state: ChatState):
        question = contextualize(state.get("user_message", ""), state.get("messages", []))
        actor = actor_from_state(state)
        settings = get_settings()
        base = {"retrieval_question": question, "citations": [], "task_result": "", "grounded": False}
        original = state.get("user_message", "")
        enterprise_intent = looks_enterprise(original)
        # 非 OA 制度问题先尝试企业知识库；明确的企业问题即使无命中也不混用 OA 语料。
        if enterprise_intent or not looks_oa_policy(original):
            from enterprise_rag import config as rag_config
            from enterprise_rag.retrieval import retrieve as enterprise_retrieve
            result = enterprise_retrieve(question, user=actor, top_k=settings.kb_top_k)
            if result.get("chunks") and result["debug"].get("top_score", 0) >= rag_config.MIN_SCORE:
                return {**base, "retrieved_chunks": result["chunks"], "knowledge_namespace": "enterprise",
                        "retrieval_meta": result["debug"]}
            if enterprise_intent:
                return {**base, "retrieved_chunks": [], "knowledge_namespace": "enterprise",
                        "retrieval_meta": result.get("debug", {})}
        candidates = retrieve(question, actor, top_k=settings.kb_top_k)
        chunks = [c for c in candidates if c.get("score", 0) >= settings.kb_min_score]
        inc("retrieval_hits_total" if chunks else "retrieval_empty_total")
        return {**base, "retrieved_chunks": chunks, "knowledge_namespace": "oa", "retrieval_meta": {}}

    async def answer_node(state: ChatState):
        chunks = state.get("retrieved_chunks") or []
        question = state.get("retrieval_question") or state.get("user_message", "")
        method = "refusal"
        if not chunks:
            text, citations, grounded = REFUSAL_TEXT, [], False
        elif state.get("knowledge_namespace") == "enterprise":
            from enterprise_rag.answer import generate_answer
            from enterprise_rag.citations import to_citations as enterprise_citations, source_summary
            generated = await asyncio.to_thread(generate_answer, question, chunks, None,
                state.get("retrieval_meta", {}).get("evidence_budget_complete", True))
            citations = enterprise_citations(generated["citations"])
            text = generated["answer"]
            if generated["citations"]:
                text += "\n\n" + source_summary(generated["citations"])
            grounded, method = generated["grounded"], generated["method"]
        else:
            answer, cited, grounded = REFUSAL_TEXT, [], False
            base_prompt = build_prompt(question, chunks)
            for attempt in range(2):
                prompt = base_prompt
                if attempt:
                    prompt += (
                        "\n\n严格要求：每个事实句末尾必须带 [编号]，编号只能来自【资料】；"
                        "没有引用编号的回答无效。"
                    )
                try:
                    response = await llm.ainvoke(prompt)
                    raw = response.content if hasattr(response, "content") else str(response)
                except Exception as exc:
                    logger.warning("知识问答生成失败（第 %s 次）: %s", attempt + 1, type(exc).__name__)
                    raw = REFUSAL_TEXT
                answer, cited, grounded = verify_answer(raw, chunks)
                if grounded:
                    break
            citations, text = to_citations(cited), format_with_sources(answer, cited)
            method = "llm_verified" if cited else "refusal"
        try:
            get_stream_writer()({"type": "token", "delta": text, "mode": "verified"})
        except RuntimeError:
            pass
        inc("policy_answers_total")
        return {"task_result": text, "response": text, "citations": citations,
                "grounded": grounded, "answer_method": method}

    return {"retrieve": retrieve_node, "answer": answer_node}
