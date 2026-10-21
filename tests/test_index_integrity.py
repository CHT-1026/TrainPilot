"""Prevent future-version leakage and incompatible embedding spaces; no external calls."""
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from enterprise_rag import config
from enterprise_rag.embedding import (
    EmbeddingIntegrityError,
    EmbeddingUnavailableError,
    embed_texts,
    hash_embed,
)
from enterprise_rag.store import RagIndex, _load_chunks


def chunk(doc_id="policy-v1", **overrides):
    row = {
        "chunk_id": f"{doc_id}#0",
        "doc_id": doc_id,
        "title": "测试制度",
        "section": "正文",
        "text": "测试制度：员工每年参加安全培训。",
        "doc_family": "test-policy",
        "version": "v1",
        "effective_date": "2024-01-01",
        "doc_updated_at": "2024-01-01",
    }
    row.update(overrides)
    return row


def stored_chunk(**overrides):
    row = chunk(
        embedding=hash_embed("测试制度：员工每年参加安全培训。"),
        embedding_dim=config.EMBEDDING_DIM,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_version=config.EMBEDDING_VERSION,
    )
    row.update(overrides)
    return row


class HistoricalVersionIntegrityTest(unittest.TestCase):
    def test_before_first_effective_date_does_not_choose_future_version(self):
        index = RagIndex([
            chunk(),
            chunk("policy-v2", version="v2", effective_date="2025-01-01", doc_updated_at="2025-01-01"),
        ])
        self.assertEqual(index.allowed_doc_ids("2023-12-31"), set())
        self.assertEqual(index.allowed_doc_ids("2024-01-01"), {"policy-v1"})
        self.assertEqual(index.allowed_doc_ids("2025-01-01"), {"policy-v2"})
        self.assertEqual(index.allowed_doc_ids(None), {"policy-v2"})

    def test_unknown_effective_date_is_not_assumed_to_be_historically_valid(self):
        index = RagIndex([
            chunk(effective_date=""),
            chunk("policy-v2", version="v2", effective_date="2025-01-01", doc_updated_at="2025-01-01"),
        ])
        self.assertEqual(index.allowed_doc_ids("2023-12-31"), set())


class VectorIndexIntegrityTest(unittest.TestCase):
    def test_empty_index_keeps_matrix_contract_and_returns_no_hits(self):
        index = RagIndex([])
        self.assertEqual(index.matrix.shape, (0, config.EMBEDDING_DIM))
        self.assertEqual(index.vector_search(np.zeros(config.EMBEDDING_DIM), 5), [])
        self.assertEqual(index.bm25_search("安全培训", 5), [])

    def test_matching_metadata_and_vectors_are_searchable(self):
        row = stored_chunk()
        index = RagIndex([row])
        hits = index.vector_search(np.asarray(row["embedding"]), 1)
        self.assertEqual(hits[0]["doc_id"], row["doc_id"])

    def test_model_dimension_and_revision_mismatch_are_rejected(self):
        for overrides in (
            {"embedding_dim": config.EMBEDDING_DIM + 1},
            {"embedding_model": "foreign-semantic-model"},
            {"embedding_version": "unknown-revision"},
            {"embedding_version": None},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(EmbeddingIntegrityError):
                    RagIndex([stored_chunk(**overrides)])

    def test_stored_vector_length_is_validated_separately_from_metadata(self):
        with self.assertRaises(EmbeddingIntegrityError):
            RagIndex([stored_chunk(embedding=[1.0, 0.0])])

    def test_non_finite_vector_cannot_poison_ranking(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(value=str(bad)):
                values = [bad] + [0.0] * (config.EMBEDDING_DIM - 1)
                with self.assertRaises(EmbeddingIntegrityError):
                    RagIndex([stored_chunk(embedding=values)])

    def test_query_vector_wrong_dimension_is_rejected_before_matrix_multiply(self):
        index = RagIndex([stored_chunk()])
        with self.assertRaises(EmbeddingIntegrityError):
            index.vector_search(np.array([1.0, 0.0]), 1)

    def test_remote_index_without_vectors_never_rebuilds_hash_vectors(self):
        with patch.object(config, "EMBEDDING_PROVIDER", "dashscope"):
            with self.assertRaises(EmbeddingIntegrityError):
                RagIndex([chunk()])


class PersistedEmbeddingIntegrityTest(unittest.TestCase):
    def load_rows(self, rows):
        fake_connection = MagicMock()
        cursor = fake_connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = rows
        with patch("enterprise_rag.store.connection") as connect:
            connect.return_value.__enter__.return_value = fake_connection
            loaded = _load_chunks("integrity-test")
        self.assertIn("c.embedding_model", cursor.execute.call_args.args[0])
        return loaded

    def database_row(self, **overrides):
        row = stored_chunk()
        row["embedding_json"] = json.dumps(row.pop("embedding"))
        row.update(overrides)
        return row

    def test_database_load_preserves_verified_embedding_contract(self):
        loaded = self.load_rows([self.database_row()])
        self.assertEqual(RagIndex(loaded).matrix.shape, (1, config.EMBEDDING_DIM))

    def test_database_missing_or_corrupt_vector_is_rejected_instead_of_rehashed(self):
        for payload in (None, "", "{invalid json", "null", "[1.0, 0.0]"):
            with self.subTest(payload=payload):
                with self.assertRaises(EmbeddingIntegrityError):
                    self.load_rows([self.database_row(embedding_json=payload)])

    def test_database_other_model_is_rejected_even_at_same_dimension(self):
        with self.assertRaises(EmbeddingIntegrityError):
            self.load_rows([self.database_row(embedding_model="foreign-semantic-model")])


class RemoteEmbeddingIntegrityTest(unittest.TestCase):
    def test_provider_outage_is_explicit_and_never_calls_hash_fallback(self):
        with patch.object(config, "EMBEDDING_PROVIDER", "dashscope"):
            with patch("enterprise_rag.embedding._dashscope_embed", side_effect=RuntimeError("simulated outage")):
                with patch("enterprise_rag.embedding.hash_embed") as fallback:
                    with self.assertRaises(EmbeddingUnavailableError):
                        embed_texts(["安全培训"])
                    fallback.assert_not_called()

    def test_remote_count_dimension_and_numeric_integrity_are_required(self):
        for response in (
            [],
            [[1.0, 0.0]],
            [[float("nan")] + [0.0] * (config.EMBEDDING_DIM - 1)],
        ):
            with self.subTest(response_size=len(response)):
                with patch.object(config, "EMBEDDING_PROVIDER", "dashscope"):
                    with patch("enterprise_rag.embedding._dashscope_embed", return_value=response):
                        with self.assertRaises(EmbeddingIntegrityError):
                            embed_texts(["安全培训"])

    def test_remote_success_remains_in_configured_vector_space(self):
        vector = [1.0] + [0.0] * (config.EMBEDDING_DIM - 1)
        with patch.object(config, "EMBEDDING_PROVIDER", "dashscope"):
            with patch("enterprise_rag.embedding._dashscope_embed", return_value=[vector]):
                self.assertEqual(embed_texts(["安全培训"]), [vector])

    def test_empty_batch_never_contacts_provider(self):
        with patch.object(config, "EMBEDDING_PROVIDER", "dashscope"):
            with patch("enterprise_rag.embedding._dashscope_embed") as remote:
                self.assertEqual(embed_texts([]), [])
                remote.assert_not_called()

    def test_hash_provider_stays_deterministic_and_normalized(self):
        with patch.object(config, "EMBEDDING_PROVIDER", "hash"):
            values = embed_texts(["安全培训"])[0]
        self.assertEqual(values, hash_embed("安全培训"))
        self.assertAlmostEqual(sum(value * value for value in values), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
