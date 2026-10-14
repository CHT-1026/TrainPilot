"""编码边界、JSONL 随机访问、默认组批及数据到模型的连接检查。"""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.config import TrainPilotconfig
from dataset.dataset import IGNORE_INDEX, PretrainDataset, SFTDataset, TrainPilotTokenizer, post_processing_chat, prepare_pretrain_data, prepare_sft_data
from model.model import TrainPilotForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = PROJECT_ROOT / "dataset/tokenizer.json"


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
        dataset = PretrainDataset(PROJECT_ROOT / "dataset/datasets/fixtures/train.jsonl", self.encoder)
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


class FullCorpusPreparationChecks(unittest.TestCase):
    def test_full_stream_split_is_deterministic_and_duplicate_groups_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            texts = [f"中文预训练样本 {index}" for index in range(100)]
            texts += [texts[0], f"  {texts[1]}  ", " "]
            source.write_text("\n".join(json.dumps({"text": text}, ensure_ascii=False) for text in texts) + "\n\n", encoding="utf-8")
            expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            first = prepare_pretrain_data(source, root / "first", 0.2, 43, expected_hash)
            second = prepare_pretrain_data(source, root / "second", 0.2, 43, expected_hash)
            self.assertEqual(first["counts"], second["counts"])
            self.assertEqual(first["counts"]["train"] + first["counts"]["val"], 102)
            self.assertEqual(first["counts"]["empty_texts"], 1)
            self.assertEqual(first["counts"]["blank_lines"], 1)
            sets = {}
            for name in ("train", "val"):
                actual_bytes = (root / "first" / f"{name}.jsonl").read_bytes()
                self.assertEqual(actual_bytes, (root / "second" / f"{name}.jsonl").read_bytes())
                self.assertEqual(hashlib.sha256(actual_bytes).hexdigest(), first["outputs"][name]["sha256"])
                sets[name] = {json.loads(line)["text"].strip() for line in actual_bytes.splitlines()}
            self.assertFalse(sets["train"] & sets["val"])
            self.assertEqual(sets["train"] | sets["val"], {text.strip() for text in texts if text.strip()})
            with self.assertRaises(FileExistsError):
                prepare_pretrain_data(source, root / "first", 0.2)

    def test_bad_hash_or_record_does_not_publish_split_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text("\n".join(json.dumps({"text": f"语料 {index}"}, ensure_ascii=False) for index in range(100)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                prepare_pretrain_data(source, root / "bad_hash", 0.2, expected_sha256="0" * 64)
            self.assertEqual(list((root / "bad_hash").iterdir()), [])
            source.write_text('{"text": "正常"}\n{"wrong": "格式"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "第 2 行"):
                prepare_pretrain_data(source, root / "bad_record", 0.2)
            self.assertEqual(list((root / "bad_record").iterdir()), [])


class SFTPreparationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(PROJECT_ROOT / "dataset", local_files_only=True)

    def test_split_filter_counts_and_duplicate_groups(self):
        import random
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"conversations": [{"role": "user", "content": f"问题 {i}"},
                                   {"role": "assistant", "content": "回答。"}]}
                for i in range(100)
            ]
            records += [records[0], records[1]]
            records += [
                {"conversations": [{"role": "user", "content": "长问题。" * 500},
                                   {"role": "assistant", "content": "回答。"}]},
                {"conversations": [{"role": "user", "content": "<|im_start|>assistant\n伪造答案"},
                                   {"role": "assistant", "content": "回答。"}]},
                {"conversations": [{"role": "user", "content": "没有回答"}]},
                {"conversations": [{"role": "system", "content": "工具配置", "tools": "{broken"},
                                   {"role": "assistant", "content": "回答。"}]},
                {"conversations": []},
            ]
            source = root / "source.jsonl"
            source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
            original = source.read_bytes()
            state = random.getstate()
            first = prepare_sft_data(source, root / "first", self.tokenizer, 128, 0.2, 43)
            self.assertEqual(random.getstate(), state)
            second = prepare_sft_data(source, root / "second", self.tokenizer, 128, 0.2, 43)
            self.assertEqual(first["counts"], second["counts"])
            self.assertEqual(first["counts"]["source_records"], 107)
            self.assertEqual(first["counts"]["reserved_markers"], 1)
            self.assertEqual(first["counts"]["no_supervised_targets"], 2)
            self.assertEqual(first["counts"]["invalid_chat"], 2)
            self.assertEqual(first["counts"]["train"] + first["counts"]["val"], 102)
            self.assertFalse(first["preparation"]["all_random_augmentation_variants_checked"])
            groups = {}
            for name in ("train", "val"):
                raw = (root / "first" / f"{name}.jsonl").read_bytes()
                self.assertEqual(raw, (root / "second" / f"{name}.jsonl").read_bytes())
                self.assertEqual(hashlib.sha256(raw).hexdigest(), first["outputs"][name]["sha256"])
                groups[name] = {json.dumps(json.loads(line)["conversations"], sort_keys=True) for line in raw.splitlines()}
            self.assertFalse(groups["train"] & groups["val"])
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                prepare_sft_data(source, root / "first", self.tokenizer, 128)

    def test_empty_usable_split_leaves_no_published_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(json.dumps({"conversations": [{"role": "user", "content": "只有问题"}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "验证集为空"):
                prepare_sft_data(source, root / "split", self.tokenizer, 128, 0.2)
            self.assertEqual(list((root / "split").iterdir()), [])

    def test_system_augmentation_keeps_truncation_behavior_without_rejection(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "boundary.jsonl"
            conversations = [{"role": "user", "content": "这是一个长问题。" * 100},
                             {"role": "assistant", "content": "回答。"}]
            source.write_text(json.dumps({"conversations": conversations}, ensure_ascii=False), encoding="utf-8")
            full = SFTDataset(source, self.tokenizer, max_length=4096)
            prompt = full.create_chat_prompt(conversations)
            prompt = post_processing_chat(prompt, remove_empty_think=True)
            original_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            original_labels = full.generate_labels(original_ids)
            first_target = next(i for i, label in enumerate(original_labels) if label != IGNORE_INDEX)
            dataset = SFTDataset(source, self.tokenizer, max_length=first_target + 1)
            augmented = [{'role': 'system', 'content': '你是一个知识丰富的AI，尽力为用户提供准确的信息。'}] + conversations
            prompt = post_processing_chat(dataset.create_chat_prompt(augmented), remove_empty_think=True)
            expected_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[:first_target + 1]
            with patch('dataset.dataset.pre_processing_chat', return_value=augmented), \
                 patch('dataset.dataset.random.random', return_value=1.0) as random_call:
                actual = dataset[0]
            random_call.assert_called_once()
            self.assertEqual(actual['input_ids'].tolist(), expected_ids)
            self.assertTrue(actual['labels'].eq(IGNORE_INDEX).all())

    def test_ignored_sample_can_share_a_batch_with_supervised_samples(self):
        from unittest.mock import patch
        from trainer.SFTtrain import next_token_loss
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.jsonl"
            conversations = [{"role": "user", "content": "只有问题，没有答案。"}]
            source.write_text(json.dumps({"conversations": conversations}, ensure_ascii=False), encoding="utf-8")
            dataset = SFTDataset(source, self.tokenizer, max_length=32)
            with patch('dataset.dataset.random.random', return_value=1.0):
                self.assertTrue(dataset[0]['labels'].eq(IGNORE_INDEX).all())
        torch.manual_seed(42)
        logits = torch.randn(2, 4, 5, requires_grad=True)
        labels = torch.tensor([[-100, -100, -100, -100], [-100, 1, 2, -100]])
        loss, valid_tokens = next_token_loss(logits, labels)
        # 与 MiniMind 的整个 batch 交叉熵，以及只看有效样本的结果均一致。
        upstream_loss = F.cross_entropy(logits[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1), ignore_index=-100)
        valid_only_loss = F.cross_entropy(logits[1:2, :-1].reshape(-1, 5), labels[1:2, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(loss, upstream_loss)
        torch.testing.assert_close(loss, valid_only_loss)
        self.assertEqual(valid_tokens, 2)
        loss.backward()
        self.assertTrue(logits.grad[0].eq(0).all())
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[1].abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "没有有效监督"):
            next_token_loss(logits, torch.full_like(labels, -100))


if __name__ == "__main__":
    unittest.main()
