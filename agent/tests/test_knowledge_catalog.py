"""咨询目录与具体问答分离，并复用当前资料权限。"""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.app.capabilities.retrieval.catalog import is_catalog_question, visible_catalog, render_catalog
from agent.app.domain.user.models import UserContext
from agent.app.workflows.chat_nodes.policy_qa import create_policy_nodes


class KnowledgeCatalogChecks(unittest.TestCase):
    def test_scope_question_is_separate_from_product_facts(self):
        self.assertTrue(is_catalog_question("我想查询公司的业务，你可以为我提供哪些方面的知识咨询"))
        self.assertTrue(is_catalog_question("知识库覆盖哪些主题？"))
        for text in ("公司主营业务是什么？", "LT-R20的负载是多少？", "知识库里有哪些机器人产品？"):
            self.assertFalse(is_catalog_question(text), text)

    def test_catalog_filters_permissions_and_deduplicates_chunks(self):
        public = dict(doc_id="product", title="产品总览", category="product", department=None, visibility="all")
        private = dict(doc_id="finance", title="受限财务资料", category="finance", department=None, visibility="finance")
        user = UserContext(user_id="employee", name="测试用户", role="employee")
        with patch("agent.app.capabilities.retrieval.catalog.get_index", side_effect=[
                SimpleNamespace(chunks=[public, public, private]), SimpleNamespace(chunks=[])]):
            examples, meta = visible_catalog(user)
        self.assertEqual(meta["documents"], 1)
        text = render_catalog(examples, meta)
        self.assertIn("产品总览", text)
        self.assertNotIn("受限财务资料", text)
        self.assertTrue(all(c["text"] == "" for c in examples))

    def test_empty_catalog_does_not_invent_topics(self):
        self.assertIn("没有可列出", render_catalog([], {"documents": 0}))

    def test_catalog_answer_does_not_ask_model_to_invent_capabilities(self):
        chunk = dict(doc_id="product", chunk_id="c1", title="产品总览", category="product", text="", score=0)
        result = asyncio.run(create_policy_nodes(None)["answer"]({
            "knowledge_namespace": "catalog", "retrieved_chunks": [chunk],
            "retrieval_meta": {"catalog": True, "documents": 1}, "user_message": "知识库覆盖哪些主题？"}))
        self.assertEqual(result["answer_method"], "knowledge_catalog")
        self.assertTrue(result["grounded"])
        self.assertIn("产品总览", result["response"])
        self.assertEqual(result["citations"][0]["doc_id"], "product")


if __name__ == "__main__":
    unittest.main()
