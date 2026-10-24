"""Embedding adapter obeys the request's remaining deadline and cancellation."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from enterprise_rag import config, embedding
from enterprise_rag.budget import RequestBudget, RequestCancelled, bind_budget


class EmbeddingDeadlineTest(unittest.TestCase):
    def test_query_call_uses_remaining_timeout_without_hidden_sdk_retries(self):
        budget=RequestBudget();client=MagicMock()
        client.embeddings.create.return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[1,0])])
        with bind_budget(budget),patch.object(budget,'remaining_seconds',side_effect=[.05,.04]), \
             patch('openai.OpenAI',return_value=client) as create:
            embedding._dashscope_embed(['query'])
        self.assertEqual(create.call_args.kwargs['max_retries'],0)
        self.assertEqual(create.call_args.kwargs['timeout'],.05)
        self.assertEqual(client.embeddings.create.call_args.kwargs['timeout'],.04)

    def test_expired_budget_never_constructs_provider(self):
        budget=RequestBudget(deadline_seconds=0)
        with bind_budget(budget),patch('openai.OpenAI') as create:
            with self.assertRaises(embedding.EmbeddingBudgetExceeded):embedding._dashscope_embed(['query'])
        create.assert_not_called()

    def test_cancel_is_not_relabelled_as_embedding_outage(self):
        budget=RequestBudget(cancel_check=lambda:True)
        with bind_budget(budget),patch.object(config,'EMBEDDING_PROVIDER','dashscope'),patch('openai.OpenAI') as create:
            with self.assertRaises(RequestCancelled):embedding.embed_texts(['query'])
        create.assert_not_called()
