"""预训练数据：文本编码、按偏移读取 JSONL、检查一个 batch。

文本编码由 TrainPilotTokenizer 完成；PretrainDataset 根据字节偏移读取 JSONL。

每条 JSONL 记录必须包含字符串字段 text。单条样本返回三个 [T] 张量；
DataLoader 默认将同名张量堆叠为 [B,T]。这里只读取数据，不进行训练。

labels 与 input_ids 保持同位置；padding 标签为 -100。
下一 token 的移位留给训练端：logits[:, :-1] 对应 labels[:, 1:]。
"""
import argparse
from array import array
import json
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer


IGNORE_INDEX = -100


class TrainPilotTokenizer:
    """读取本地词表，并明确管理 BOS/EOS、截断与右侧 padding。"""

    def __init__(self, tokenizer_path: str | Path, max_seq_len: int):
        if max_seq_len < 2:
            raise ValueError("max_seq_len 至少为 2，给 BOS 和 EOS 各留一个位置")
        self.max_seq_len = max_seq_len
        self.tokenizer_path = Path(tokenizer_path).expanduser().resolve()
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        # 所有长度处理由下面的 encode_sample 完成，避免执行两遍。
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.bos_id = self._special_id("<|im_start|>")
        self.eos_id = self._special_id("<|im_end|>")
        self.pad_id = self._special_id("<|endoftext|>")

    def _special_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        # 0 是合法编号，因此这里不能用 if not token_id。
        if token_id is None:
            raise ValueError(f"词表缺少必需的特殊 token: {token}")
        return token_id

    def encode_sample(self, text: str) -> dict[str, torch.Tensor]:
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")

        # 1. 正文编码，禁止自动加特殊 token
        body_ids = self.tokenizer.encode(text, add_special_tokens=False).ids

        # 2. 先截断正文，再加 BOS/EOS
        body_ids = body_ids[: self.max_seq_len - 2]
        valid_ids = [self.bos_id] + body_ids + [self.eos_id]
        valid_length = len(valid_ids)
        padding_length = self.max_seq_len - valid_length

        # 3. 右侧 padding
        input_ids = torch.tensor(
            valid_ids + [self.pad_id] * padding_length, dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [1] * valid_length + [0] * padding_length, dtype=torch.long,
        )

        # 4. labels 与 input_ids 同位置，padding 位置设 -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PretrainDataset(Dataset):
    """按字节偏移随机访问 JSONL，不把整份语料的文本保存在内存中。
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: TrainPilotTokenizer):
        self.jsonl_path = Path(jsonl_path).expanduser().resolve()
        self.tokenizer = tokenizer
        # Q 表示无符号 64 位整数；每条记录的偏移占 8 字节。
        self._offsets = array("Q")
        self._build_index()
        if not self._offsets:
            raise ValueError(f"数据文件没有有效记录: {self.jsonl_path}")

    def _read_text(self, raw_line: bytes, location: str) -> str:
        try:
            record = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"{self.jsonl_path}: {location} 不是有效的 JSON 文本") from error
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"{self.jsonl_path}: {location} 必须包含字符串字段 text")
        # 不隐式 strip 或过滤空字符串；空文本仍可监督 BOS -> EOS。
        return record["text"]

    def _build_index(self) -> None:
        # 使用二进制模式：seek 的字节偏移不受中文字符宽度和换行格式影响。
        with self.jsonl_path.open("rb") as source:
            line_number = 0
            while True:
                offset = source.tell()
                raw_line = source.readline()
                if not raw_line:
                    break
                line_number += 1
                if not raw_line.strip():
                    continue
                self._read_text(raw_line, f"第 {line_number} 行")
                self._offsets.append(offset)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        offset = self._offsets[index]
        with self.jsonl_path.open("rb") as source:
            source.seek(offset)
            raw_line = source.readline()
        text = self._read_text(raw_line, f"字节偏移 {offset}")
        return self.tokenizer.encode_sample(text)


def prepare_pretrain_data(
    source_path: str | Path,
    output_dir: str | Path,
    val_fraction: float = 0.01,
    seed: int = 43,
    expected_sha256: str | None = None,
) -> dict:
    """流式划分完整 JSONL，归一化后相同文本必定进入同一个集合。

    通过 seed 与 strip 后文本的 SHA256 分桶，不把全文装进内存。
    不移除集合内重复文本，不检查近似重复；验证比例是近似值。
    原文写入输出文件，strip 仅用于划分；空白文本跳过。
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction 必须在 0 与 1 之间")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed 必须为非负整数")
    source_path = Path(source_path).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: output_dir / f"{name}.jsonl" for name in ("train", "val")}
    manifest_path = output_dir / "manifest.json"
    if any(path.exists() for path in (*targets.values(), manifest_path)):
        raise FileExistsError("输出目录已有训练/验证数据或 manifest，请使用新的目录")
    counts = {"source_lines": 0, "blank_lines": 0, "empty_texts": 0, "train": 0, "val": 0}
    source_digest = hashlib.sha256()
    output_digests = {name: hashlib.sha256() for name in targets}
    threshold = int(val_fraction * (1 << 64))
    temporary_paths = {}
    temporary_files = {}
    try:
        for name in targets:
            temporary = NamedTemporaryFile(dir=output_dir, prefix=f".{name}.", suffix=".tmp", delete=False)
            temporary_paths[name] = Path(temporary.name)
            temporary_files[name] = temporary
        with source_path.open("rb") as source:
            for raw_line in source:
                source_digest.update(raw_line)
                counts["source_lines"] += 1
                if not raw_line.strip():
                    counts["blank_lines"] += 1
                    continue
                try:
                    record = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise ValueError(f"语料第 {counts['source_lines']} 行不是有效 JSON") from error
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise ValueError(f"语料第 {counts['source_lines']} 行缺少字符串 text")
                text = record["text"]
                normalized = text.strip()
                if not normalized:
                    counts["empty_texts"] += 1
                    continue
                bucket_hash = hashlib.sha256(f"{seed}\0{normalized}".encode("utf-8")).digest()
                name = "val" if int.from_bytes(bucket_hash[:8], "big") < threshold else "train"
                encoded = (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")
                temporary_files[name].write(encoded)
                output_digests[name].update(encoded)
                counts[name] += 1
        actual_sha256 = source_digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
            raise ValueError("原始语料 SHA256 与预期不一致，未发布划分结果")
        if counts["train"] == 0 or counts["val"] == 0:
            raise ValueError("划分后训练集或验证集为空，请增加数据量或调整比例")
        for name, temporary in temporary_files.items():
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary.close()
            os.replace(temporary_paths[name], targets[name])
        manifest = {
            "source": {"path": str(source_path), "bytes": source_path.stat().st_size, "sha256": actual_sha256},
            "preparation": {
                "split": "sha256_bucket_by_seed_and_stripped_text", "seed": seed,
                "val_fraction_requested": val_fraction, "within_split_deduplicated": False,
                "exact_normalized_text_groups_disjoint": True, "near_duplicate_checked": False,
                "tokenized_overlap_checked": False,
            },
            "counts": counts,
            "outputs": {name: {"path": str(path), "sha256": output_digests[name].hexdigest()} for name, path in targets.items()},
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest
    finally:
        for temporary in temporary_files.values():
            temporary.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
