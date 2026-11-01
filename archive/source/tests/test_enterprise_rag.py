"""企业 RAG 纯函数单元测试：查询改写、时间抽取、切分、向量与版本消解。"""
import unittest

from enterprise_rag import config
from enterprise_rag.chunking import chunk_document, _split_table
from enterprise_rag.embedding import hash_embed
from enterprise_rag.query import extract_time, rewrite_query
from enterprise_rag.store import RagIndex
from app.workflows.chat_nodes.enterprise_scope import looks_enterprise, looks_oa_policy


class QueryRewriteTest(unittest.TestCase):
    def test_month_time(self):
        self.assertEqual(extract_time("2025年7月修订后标准")[0], "2025-07-31")

    def test_quarter_time(self):
        self.assertEqual(extract_time("2025年第三季度营收")[0], "2025-09-30")

    def test_year_time(self):
        self.assertEqual(extract_time("2024年差旅标准")[0], "2024-12-31")

    def test_synonyms(self):
        rewritten = rewrite_query("出差住宿能报销多少")
        self.assertIn("差旅", rewritten["expansions"])
        self.assertTrue(rewritten["rewritten"].startswith("出差住宿能报销多少"))

    def test_possessive_noise_filtered(self):
        from app.capabilities.retrieval.tokenizer import tokenize
        from enterprise_rag.query import query_terms

        terms = query_terms("P1 故障的响应时间", "P1 故障的响应时间")
        self.assertNotIn("障的", terms)
        self.assertIn("p1", terms)
        self.assertTrue(all(len(term) >= 2 for term in tokenize("测试") if not term.isascii()))


class ChunkingTest(unittest.TestCase):
    def test_text_chunks_have_title_prefix(self):
        document = {"doc_id": "d1", "title": "测试制度"}
        blocks = [{"section": "一、范围", "text": "第一条。第二条。", "type": "text"}]
        chunks = chunk_document(document, blocks)
        self.assertTrue(chunks)
        self.assertTrue(chunks[0]["text"].startswith("测试制度 · 一、范围"))
        self.assertEqual(chunks[0]["chunk_type"], "text")

    def test_table_repeats_header(self):
        header = "| 型号 | 负载 |"
        rows = "\n".join(f"| LT-R{i} | {i} kg |" for i in range(1, 13))
        pieces = _split_table(f"{header}\n{rows}")
        self.assertEqual(len(pieces), 2)
        self.assertTrue(all(piece.startswith(header) for piece in pieces))


class EmbeddingTest(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        first = hash_embed("澜图智能 工业机器人")
        second = hash_embed("澜图智能 工业机器人")
        self.assertEqual(first, second)
        norm = sum(value * value for value in first) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=4)
        self.assertEqual(len(first), config.EMBEDDING_DIM)


class ScopeTest(unittest.TestCase):
    def test_company_identity_is_enterprise(self):
        self.assertTrue(looks_enterprise("公司叫什么名字？"))
        self.assertTrue(looks_enterprise("澜图智能是一家什么公司？"))
        self.assertTrue(looks_enterprise("公司总部在哪里？"))

    def test_leave_policy_keeps_oa_scope(self):
        self.assertFalse(looks_enterprise("年假有几天？"))
        self.assertFalse(looks_enterprise("公司食堂几点开门？"))
        self.assertTrue(looks_oa_policy("社招和校招的试用期分别是多久？"))


class VersionResolutionTest(unittest.TestCase):
    def _index(self):
        rows = []
        versions = [
            ("d_v1", "travel_expense", "v1.0", "2024-03-01", "2024-03-01T09:00:00", 450),
            ("d_v2", "travel_expense", "v2.0", "2025-01-01", "2025-01-01T09:00:00", 550),
            ("d_v3", "travel_expense", "v3.0", "2025-07-01", "2025-07-01T09:00:00", 600),
        ]
        for order, (doc_id, family, version, effective, updated, value) in enumerate(versions):
            rows.append(
                {
                    "chunk_id": f"{doc_id}#c0",
                    "doc_id": doc_id,
                    "title": f"差旅标准 {version}",
                    "section": "住宿",
                    "chunk_type": "text",
                    "ord": 0,
                    "text": f"住宿标准 {value} 元/晚",
                    "doc_family": family,
                    "version": version,
                    "effective_date": effective,
                    "doc_updated_at": updated,
                    "category": "policy",
                    "format": "md",
                }
            )
        return RagIndex(rows)

    def test_latest_version_default(self):
        index = self._index()
        allowed = index.allowed_doc_ids(None)
        self.assertIn("d_v3", allowed)
        self.assertNotIn("d_v1", allowed)

    def test_as_of_uses_effective_version(self):
        index = self._index()
        self.assertIn("d_v1", index.allowed_doc_ids("2024-12-31"))
        self.assertIn("d_v2", index.allowed_doc_ids("2025-06-30"))
        self.assertIn("d_v3", index.allowed_doc_ids("2025-08-01"))


if __name__ == "__main__":
    unittest.main()
