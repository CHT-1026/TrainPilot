"""编码边界、JSONL 随机访问、默认组批及数据到模型的连接检查。"""
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trainpilot.config import TrainPilotconfig
from trainpilot.data import IGNORE_INDEX, PretrainDataset, TrainPilotTokenizer
from trainpilot.model import TrainPilotForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = PROJECT_ROOT / "assets/tokenizer.json"


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.encoder = TrainPilotTokenizer(TOKENIZER_PATH, max_seq_len=32)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.jsonl_path = Path(self.temp.name) / "sample.jsonl"

    def write_records(self, records):
        self.jsonl_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
            encoding="utf-8",
        )

    def test_special_ids_and_vocabulary(self):
        self.assertEqual(self.encoder.vocab_size, 6400)
        self.assertEqual((self.encoder.bos_id, self.encoder.eos_id, self.encoder.pad_id), (1, 2, 0))

    def test_short_text_alignment_padding_and_independent_labels(self):
        text = "模型通过训练学习预测下一个 token。"
        sample = self.encoder.encode_sample(text)
        body = self.encoder.tokenizer.encode(text, add_special_tokens=False).ids
        expected_ids = [1] + body + [2]
        length = len(expected_ids)
        self.assertEqual(sample["input_ids"][:length].tolist(), expected_ids)
        self.assertEqual(sample["labels"][:length].tolist(), expected_ids)
        self.assertEqual(sample["attention_mask"].tolist(), [1] * length + [0] * (32 - length))
        self.assertTrue((sample["input_ids"][length:] == 0).all())
        self.assertTrue((sample["labels"][length:] == IGNORE_INDEX).all())
        for tensor in sample.values():
            self.assertEqual(tuple(tensor.shape), (32,))
            self.assertEqual(tensor.dtype, torch.long)
            self.assertEqual(tensor.device.type, "cpu")
        sample["labels"][0] = IGNORE_INDEX
        self.assertEqual(sample["input_ids"][0].item(), 1)

    def test_truncation_reserves_eos(self):
        encoder = TrainPilotTokenizer(TOKENIZER_PATH, max_seq_len=6)
        text = "我们记录训练配置，并检查输入与标签的对齐。" * 5
        body = encoder.tokenizer.encode(text, add_special_tokens=False).ids
        sample = encoder.encode_sample(text)
        self.assertEqual(sample["input_ids"].tolist(), [1] + body[:4] + [2])
        self.assertEqual(sample["attention_mask"].tolist(), [1] * 6)
        torch.testing.assert_close(sample["labels"], sample["input_ids"])

    def test_empty_text_and_minimum_length(self):
        sample = self.encoder.encode_sample("")
        self.assertEqual(sample["input_ids"].tolist(), [1, 2] + [0] * 30)
        self.assertEqual(sample["labels"].tolist(), [1, 2] + [-100] * 30)
        self.assertEqual((sample["labels"][1:] != IGNORE_INDEX).sum().item(), 1)
        minimal = TrainPilotTokenizer(TOKENIZER_PATH, max_seq_len=2).encode_sample("正文被截断")
        self.assertEqual(minimal["input_ids"].tolist(), [1, 2])
        with self.assertRaises(ValueError):
            TrainPilotTokenizer(TOKENIZER_PATH, max_seq_len=1)
        with self.assertRaises(TypeError):
            self.encoder.encode_sample(123)

    def test_literal_pad_token_is_not_treated_as_padding(self):
        sample = self.encoder.encode_sample("<|endoftext|>")
        self.assertEqual(sample["input_ids"][:3].tolist(), [1, 0, 2])
        self.assertEqual(sample["attention_mask"][:3].tolist(), [1, 1, 1])
        self.assertEqual(sample["labels"][:3].tolist(), [1, 0, 2])
        self.assertEqual(sample["labels"][3].item(), IGNORE_INDEX)

    def test_jsonl_unicode_offsets_blank_lines_and_random_access(self):
        texts = ["中文字符", "", "最后一行没有换行"]
        lines = [json.dumps({"text": text}, ensure_ascii=False) for text in texts]
        self.jsonl_path.write_bytes((lines[0] + "\r\n\r\n" + lines[1] + "\r\n" + lines[2]).encode("utf-8"))
        dataset = PretrainDataset(self.jsonl_path, self.encoder)
        self.assertEqual(len(dataset), 3)
        for index in (2, 0, 1, -1):
            actual = dataset[index]
            expected = self.encoder.encode_sample(texts[index])
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name])
        for index in (3, -4):
            with self.assertRaises(IndexError):
                dataset[index]

    def test_invalid_json_records_and_empty_file(self):
        self.jsonl_path.write_text('{"text":"有效"}\n{broken}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "第 2 行"):
            PretrainDataset(self.jsonl_path, self.encoder)
        for record in ({"other": "missing text"}, {"text": 123}, []):
            with self.subTest(record=record):
                self.write_records([record])
                with self.assertRaisesRegex(ValueError, "字符串字段 text"):
                    PretrainDataset(self.jsonl_path, self.encoder)
        self.jsonl_path.write_text("\n \n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "没有有效记录"):
            PretrainDataset(self.jsonl_path, self.encoder)

    def test_dataloader_groups_dictionaries_and_keeps_last_batch(self):
        self.write_records([{"text": text} for text in ("第一条", "第二条", "第三条")])
        dataset = PretrainDataset(self.jsonl_path, self.encoder)
        batches = list(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0))
        self.assertEqual(len(batches), 2)
        for name in ("input_ids", "attention_mask", "labels"):
            self.assertEqual(tuple(batches[0][name].shape), (2, 32))
            self.assertEqual(tuple(batches[1][name].shape), (1, 32))
            torch.testing.assert_close(batches[0][name][0], dataset[0][name])

    def test_real_tokenizer_batch_connects_to_model_and_loss(self):
        torch.manual_seed(42)
        dataset = PretrainDataset(PROJECT_ROOT / "tests/fixtures/train.jsonl", self.encoder)
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        model = TrainPilotForCausalLM(TrainPilotconfig(
            vocab_size=self.encoder.vocab_size, hidden_size=16, num_layers=1,
            num_heads=4, num_kv_heads=2, intermediate_size=32, max_seq_len=32,
        ))
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.encoder.vocab_size),
            batch["labels"][:, 1:].reshape(-1), ignore_index=IGNORE_INDEX,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.model.embed_tokens.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
