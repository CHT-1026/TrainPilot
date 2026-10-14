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
import random
from datasets import load_dataset, Features, Value
from tqdm.auto import tqdm

IGNORE_INDEX = -100

class TrainPilotTokenizer:

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


# 预训练的数据处理
def prepare_pretrain_data(
    source_path: str | Path,
    output_dir: str | Path,
    val_fraction: float = 0.01,
    seed: int = 43,
    expected_sha256: str | None = None,
) -> dict:
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

# 预训练数据集对象
class PretrainDataset(Dataset):

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


# SFT的数据处理（添加system信息）,如果对话中有工具调用，则不进行任何处理直接返回，因为涉及到工具的调用一般是固定格式的，如果随意添加system可能会出现意料外的错误。如果对话的开始没有system标识，则概率添加system标识
def pre_processing_chat(conversations, add_system_ratio=0.2):
    if not conversations:
        raise ValueError("conversations 不能为空")
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

# 数据处理：以 80% 概率删除空 think，以 20% 概率保留。
def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

# SFT数据集对象
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, *, cache_dir=None):
        super().__init__()
        if type(max_length) is not int or max_length < 2:
            raise ValueError("max_length 必须是至少为 2 的整数")
        if any(token_id is None for token_id in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)):
            raise ValueError("tokenizer 必须配置 BOS、EOS 和 PAD")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parent / 'datasets' / 'cache'
        # 从huggingface解析对象的参数
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=str(jsonl_path), split='train', features=features, cache_dir=str(self.cache_dir))
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)
    # 对读取数据的处理，将工具列表和工具调用列表解析成为对象，并且返回拼接好的消息列表
    def create_chat_prompt(self, conversations):
        if not conversations:
            raise ValueError("conversations 不能为空")
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get('role') not in {'system', 'user', 'assistant', 'tool'}:
                raise ValueError(f"不支持的角色: {message.get('role')}")
            # 此版通过角色标记定位答案。正文不能伪造模板的边界标记。
            # tools/tool_calls 中的字符串也会进入模板，所以一并检查。
            for field in ('content', 'reasoning_content', 'tools', 'tool_calls'):
                value = message.get(field)
                if value is None:
                    continue
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                if any(marker in text for marker in (self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token)):
                    raise ValueError(f"{field} 含保留的角色或 padding 标记，请清洗该样本")
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    # BOS 本身由各个角色共用；这里匹配的是 BOS + assistant + 换行。
    # labels 保持同位置，只有 assistant 正文及结束标记参与监督，不在此移位。
    def generate_labels(self, input_ids, attention_mask=None):
        # HF 的特殊 token ID 是动态属性；一次取出，避免每个 token 都查询。
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        bos_token_id = self.tokenizer.bos_token_id
        labels = [IGNORE_INDEX] * len(input_ids)
        if attention_mask is None:
            # 兼容单独调用本函数：只将右侧连续 PAD 视为 padding。
            valid_length = len(input_ids)
            while valid_length and input_ids[valid_length - 1] == pad_token_id:
                valid_length -= 1
            attention_mask = [1] * valid_length + [0] * (len(input_ids) - valid_length)
        else:
            if len(attention_mask) != len(input_ids):
                raise ValueError("attention_mask 与 input_ids 长度必须一致")
            valid_length = sum(attention_mask)
            if list(attention_mask) != [1] * valid_length + [0] * (len(input_ids) - valid_length):
                raise ValueError("SFTDataset 只支持右侧 padding 的 0/1 attention_mask")
        valid_length = min(valid_length, self.max_length)
        i = 0
        while i < valid_length:
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < valid_length:
                    if input_ids[end] == eos_token_id:
                        break
                    if input_ids[end] == bos_token_id:
                        raise ValueError("assistant 消息缺少结束标记，不能跨角色监督")
                    end += 1
                stop = end
                if end < valid_length:
                    # EOS 总要监督；若紧随模板换行，也保留该换行的监督。
                    stop = end + 1
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        stop = min(end + len(self.eos_id), valid_length)
                # 答案可能在长度上限处截断，此时仅监督已保留的有效前缀。
                for j in range(start, stop):
                    labels[j] = input_ids[j]
                i = stop
            else:
                i += 1
        return labels

    # 取出指定 index 处的样本；格式与现有 train_step/evaluate 一致。
    def __getitem__(self, index):
        # 加载数据
        sample = self.samples[index]
        # 保留原始随机增强：概率添加 system，工具数据不添加。
        conversations = pre_processing_chat(sample['conversations'])
        # 解析读取的数据并且进行拼接
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        # 模板已含特殊 token，不再自动插入；固定使用右侧截断、右侧 padding。
        input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[:self.max_length]
        valid_length = len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        attention_mask = [1] * valid_length + [0] * (self.max_length - valid_length)
        labels = self.generate_labels(input_ids, attention_mask)
        if not any(label != IGNORE_INDEX for label in labels[1:]):
            raise ValueError(f"样本 {index} 截断后没有 assistant 监督目标，请在训练前过滤或调整 max_length")
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


# 防止训练数据泄露与出现无效的数据并且根据数据集生成标准训练数据集和验证数据集
def prepare_sft_data(
    source_path: str | Path,
    output_dir: str | Path,
    tokenizer,
    max_length: int,
    val_fraction: float = 0.01,
    seed: int = 43,
) -> dict:
    """过滤不能用于当前 SFT 实现的对话，并固定划分训练/验证集。
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
        raise FileExistsError("输出目录已有划分结果，请使用新的目录")

    # HF 读取结果为 Arrow 数据集；遍历时复用现有模板与标签函数。
    dataset = SFTDataset(source_path, tokenizer, max_length=max_length)
    counts = {
        "source_records": len(dataset), "invalid_chat": 0,
        "reserved_markers": 0, "no_supervised_targets": 0,
        "train": 0, "val": 0,
    }
    reserved = (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
    threshold = int(val_fraction * (1 << 64))
    temporary_paths = {}
    temporary_files = {}
    digests = {name: hashlib.sha256() for name in targets}
    try:
        for name in targets:
            temporary = NamedTemporaryFile(dir=output_dir, prefix=f".{name}.", suffix=".tmp", delete=False)
            temporary_paths[name] = Path(temporary.name)
            temporary_files[name] = temporary

        for sample in tqdm(dataset.samples, total=len(dataset), desc="SFT 模板编码、过滤与划分", unit="条"):
            conversations = sample["conversations"]
            if not conversations or any(not isinstance(message, dict) for message in conversations):
                counts["invalid_chat"] += 1
                continue
            # 角色字段之外的内容会进入模板；不删除标记来篡改原始答案。
            content_json = json.dumps(
                [{key: value for key, value in message.items() if key != "role"}
                 for message in conversations], ensure_ascii=False,
            )
            if any(marker in content_json for marker in reserved):
                counts["reserved_markers"] += 1
                continue
            try:
                prompt = dataset.create_chat_prompt(conversations)
                prompt = post_processing_chat(prompt, remove_empty_think=True)
                input_ids = tokenizer(prompt, add_special_tokens=False).input_ids[:max_length]
                labels = dataset.generate_labels(input_ids, [1] * len(input_ids))
            except (ValueError, TypeError):
                # JSON 格式的 tools/tool_calls 解析失败也属于此类。
                counts["invalid_chat"] += 1
                continue
            if not any(label != IGNORE_INDEX for label in labels[1:]):
                counts["no_supervised_targets"] += 1
                continue

            # 哈希仅用于分组，输出保留原文和工具字段，不执行近似去重。
            group = json.dumps(conversations, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            bucket = hashlib.sha256(f"{seed}\0{group}".encode("utf-8")).digest()
            name = "val" if int.from_bytes(bucket[:8], "big") < threshold else "train"
            encoded = (json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n").encode("utf-8")
            temporary_files[name].write(encoded)
            digests[name].update(encoded)
            counts[name] += 1

        if counts["train"] == 0 or counts["val"] == 0:
            raise ValueError("过滤后训练集或验证集为空，请增加数据量或调整验证比例")
        for name, temporary in temporary_files.items():
            temporary.close()
            os.replace(temporary_paths[name], targets[name])
        manifest = {
            "source": str(source_path),
            "preparation": {
                "max_length": max_length, "seed": seed,
                "val_fraction_requested": val_fraction,
                "split": "sha256_bucket_by_seed_and_canonical_conversations",
                "exact_conversation_groups_disjoint": True,
                "within_split_deduplicated": False,
                "supervision_check": "unaugmented_chat_with_empty_think_removed",
                "all_random_augmentation_variants_checked": False,
            },
            "counts": counts,
            "outputs": {
                name: {"path": str(path), "sha256": digests[name].hexdigest()}
                for name, path in targets.items()
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest
    finally:
        for temporary in temporary_files.values():
            temporary.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
