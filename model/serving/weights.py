"""Shared read-only Full-SFT weight loader and deterministic KV-cache generation."""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import gc
import hashlib
import io
import math
import pickle
import re
import time
import zipfile
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class StorageRef:
    dtype: object
    key: str
    size: int


@dataclass
class TensorRef:
    storage: StorageRef
    offset: int
    shape: tuple
    stride: tuple


def rebuild_tensor_ref(storage, offset, shape, stride, *unused):
    return TensorRef(storage, offset, tuple(shape), tuple(stride))


class CheckpointMetadata(pickle.Unpickler):
    """只解析 torch.save 张量引用；不执行任意 pickle 全局对象，不分配优化器存储。"""
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module == "torch._utils" and name in {"_rebuild_tensor", "_rebuild_tensor_v2"}:
            return rebuild_tensor_ref
        storage_types = {"FloatStorage":torch.float32, "DoubleStorage":torch.float64,
                         "HalfStorage":torch.float16, "BFloat16Storage":torch.bfloat16,
                         "ByteStorage":torch.uint8, "CharStorage":torch.int8,
                         "ShortStorage":torch.int16, "IntStorage":torch.int32,
                         "LongStorage":torch.int64, "BoolStorage":torch.bool}
        if module == "torch" and name in storage_types:
            return storage_types[name]
        raise pickle.UnpicklingError(f"不支持的 checkpoint 全局对象：{module}.{name}")

    def persistent_load(self, identifier):
        kind, dtype, key, location, size = identifier
        if kind != "storage" or not isinstance(dtype, torch.dtype):
            raise pickle.UnpicklingError("无效的张量存储引用")
        return StorageRef(dtype, str(key), size)


def tensor_chunks(archive, prefix, ref):
    expected_stride = 1
    for size, stride in zip(reversed(ref.shape), reversed(ref.stride)):
        if size > 1 and stride != expected_stride:
            raise ValueError("流式权重加载仅接受连续张量")
        expected_stride *= size
    element_size = torch.empty((), dtype=ref.storage.dtype).element_size()
    count = math.prod(ref.shape)
    if ref.offset < 0 or ref.offset + count > ref.storage.size:
        raise ValueError("张量视图超出存储边界")
    with archive.open(f"{prefix}data/{ref.storage.key}") as stream:
        stream.seek(ref.offset * element_size)
        left = count * element_size
        while left:
            chunk = stream.read(min(left, 16 * 1024**2))
            if not chunk or len(chunk) % element_size:
                raise EOFError("checkpoint 张量内容不完整")
            yield chunk
            left -= len(chunk)


def read_tensor(archive, prefix, ref, device, dtype):
    target = torch.empty(ref.shape, device=device, dtype=dtype)
    flat, offset = target.view(-1), 0
    for chunk in tensor_chunks(archive, prefix, ref):
        source = torch.frombuffer(bytearray(chunk), dtype=ref.storage.dtype)
        # 先在 CPU 转换，避免大矩阵以 FP32 临时落到显卡；每次只处理16MiB原始数据。
        flat[offset:offset + source.numel()].copy_(source.to(dtype=dtype))
        offset += source.numel()
    return target


def tensor_hash(archive, prefix, ref):
    digest = hashlib.sha256()
    for chunk in tensor_chunks(archive, prefix, ref):
        digest.update(chunk)
    return digest.digest()


def load_model(checkpoint, config_dir, device, dtype, *, reserve_bytes=1024**3):
    archive = zipfile.ZipFile(checkpoint)
    member = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
    prefix = member[:-len("data.pkl")]
    state = CheckpointMetadata(io.BytesIO(archive.read(member))).load()
    config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype, attn_implementation="sdpa")
    if torch.device(device).type == "cuda":
        device_index = torch.device(device).index
        free, _ = torch.cuda.mem_get_info(torch.cuda.current_device() if device_index is None else device_index)
        required = sum(p.numel()*p.element_size() for p in model.parameters()) + reserve_bytes
        if free < required:
            raise RuntimeError(f"可用显存不足：当前 {free/1024**3:.2f} GiB，权重加评测预留约需 {required/1024**3:.2f} GiB；请先释放其他任务的显存。")
    weights = {}
    for name, tensor in state["weights"].items():
        if name == "lm_head.weight" and config.tie_word_embeddings:
            embedding = state["weights"]["model.embed_tokens.weight"]
            same_storage = tensor.storage.key == embedding.storage.key and tensor.offset == embedding.offset
            if tensor.shape != embedding.shape or (not same_storage and tensor_hash(archive, prefix, tensor) != tensor_hash(archive, prefix, embedding)):
                raise ValueError("checkpoint 的共享嵌入与 lm_head 不一致")
            continue
        weights[name] = read_tensor(archive, prefix, tensor, device, dtype)
    archive.close()
    if config.tie_word_embeddings:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    model.load_state_dict(weights, strict=True, assign=True)
    model.tie_weights()
    # RoPE 的频率是非持久 buffer，不在 state_dict 中；meta 构造后需按配置重建。
    model.model.rotary_emb = type(model.model.rotary_emb)(config, device=torch.device(device))
    model.eval()
    metadata = {"progress": state["progress"], "signature": state["signature"],
                "unique_parameters": sum(p.numel() for p in model.parameters()),
                "config": config.to_dict()}
    del weights, state
    gc.collect()
    return model, metadata


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def answer_text(raw):
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


@torch.inference_mode()
def generate(model, tokenizer, ids, device, max_new_tokens=256, stop_at_eos=True):
    tokens = torch.tensor([ids], device=device)
    # Base 的 EOS 常为 endoftext；SFT 聊天回合实际以 im_end 结束，两者都应停止。
    eos = {x for x in (tokenizer.eos_token_id, model.config.eos_token_id,
                       tokenizer.convert_tokens_to_ids("<|im_end|>")) if x is not None}
    cache, emitted, first = None, [], None
    sync(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        outputs = model(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
        scores = outputs.logits[0, -1].float()
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError("生成 logits 出现非有限值")
        chosen = int(scores.argmax().item())
        emitted.append(chosen)
        cache = outputs.past_key_values
        del outputs, scores
        if first is None:
            sync(device)
            first = time.perf_counter() - start
        if stop_at_eos and chosen in eos:
            reason = "eos"
            break
        tokens = torch.tensor([[chosen]], device=device)
    sync(device)
    elapsed = time.perf_counter() - start
    grams = [tuple(emitted[i:i+4]) for i in range(max(0, len(emitted)-3))]
    result = {"raw_text": tokenizer.decode(emitted, skip_special_tokens=False),
              "generated_ids": emitted,
              "text": tokenizer.decode(emitted, skip_special_tokens=True),
              "input_tokens":len(ids), "generated_tokens":len(emitted), "stop_reason":reason,
              "seconds":elapsed, "first_token_seconds":first, "tokens_per_second":len(emitted)/elapsed,
              "decode_tokens_per_second": (len(emitted)-1)/(elapsed-first) if len(emitted)>1 else None,
              "repeated_4gram_fraction":1-len(set(grams))/len(grams) if grams else 0}
    if torch.device(device).type == "cuda":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device)/1024**2
    del cache
    return result
