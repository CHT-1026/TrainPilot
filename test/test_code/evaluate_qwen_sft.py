"""本地 Qwen SFT 只读评测；原始回答与数值记录在 runs，正式报告在 test/test_log。

python test/test_code/evaluate_qwen_sft.py
使用 checkpoint 中的 weights，不恢复优化器，也不下载原始 Base 权重。
"""
import argparse
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime
import gc
import hashlib
import json
import io
import math
from pathlib import Path
import platform
import pickle
import random
import re
import statistics
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import datasets  # Windows 先加载 Arrow。
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from dataset.dataset import QwenSFTEncoder
from trainer.SFTtrain import next_token_loss


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "查询指定城市当前天气",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "计算数学表达式",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def load_model(checkpoint, config_dir, device, dtype):
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
        required = sum(p.numel()*p.element_size() for p in model.parameters()) + 1024**3
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


def tool_calls(raw):
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", raw, flags=re.S)
    try:
        calls = [json.loads(block) for block in blocks]
    except json.JSONDecodeError:
        return [], False
    valid = len(blocks) == raw.count("<tool_call>") == raw.count("</tool_call>")
    for call in calls:
        valid = valid and isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict)
    return calls, bool(valid)


def grade(case, raw):
    """评分不调用外部模型，不修复输出 JSON；格式不符也会计为未通过。"""
    text = answer_text(raw)
    kind, expected = case["kind"], case.get("expected")
    if kind == "open":
        return None
    if kind == "number":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        last = lines[-1].replace("**", "").replace("，", ",")
        match = re.fullmatch(r"(?:答案|结果|最终答案)?\s*[:：=]?\s*(-?\d+(?:\.\d+)?)\s*[。.]?", last)
        return bool(match) and math.isclose(float(match.group(1)), expected, abs_tol=1e-6)
    if kind == "exact":
        return text == expected
    if kind == "contains":
        return expected in text and "<tool_call>" not in raw
    if kind == "json":
        try:
            actual = json.loads(text)
            return json.dumps(actual, sort_keys=True, ensure_ascii=False) == json.dumps(expected, sort_keys=True, ensure_ascii=False)
        except json.JSONDecodeError:
            return False
    calls, valid = tool_calls(raw)
    if kind == "no_tool":
        return bool(text) and valid and not calls and "<tool_call" not in raw
    if kind == "tool":
        return valid and len(calls) == 1 and calls[0] == expected
    if kind == "clarify":
        return bool(text) and valid and not calls and "<tool_call" not in raw and any(x in text for x in ("哪个城市", "哪座城市", "城市名称", "什么城市", "所在城市", "哪一个城市"))
    raise ValueError(kind)


def build_cases():
    cases = []
    def add(group, prompt, kind, expected=None, **extra):
        cases.append(dict(id=f"{group}-{sum(c['group'] == group for c in cases)+1:02d}",
                          group=group, prompt=prompt, kind=kind, expected=expected, **extra))
    rng = random.Random(20261019)
    for op in ("+", "-", "×"):
        for _ in range(4):
            a, b = rng.randint(12, 99), rng.randint(2, 12)
            expected = {"+": a+b, "-": a-b, "×": a*b}[op]
            add("arithmetic", f"计算 {a}{op}{b}。只输出最终数字，不要其他内容。", "number", expected)
    for prompt, expected in [
        ("小明原有3个苹果，又买了2个，再吃掉1个，还剩几个？", 4),
        ("商品原价80元，打八折后再减10元，实付多少元？", 54),
        ("每盒有6支笔，买7盒后送出9支，还剩几支？", 33),
        ("一辆车每小时60千米，行驶2.5小时，行驶多少千米？", 150),
        ("甲有120元，拿出四分之一后又收到15元，现在有多少元？", 105),
        ("一个班40人，其中女生占百分之四十五，男生有多少人？", 22),
        ("一件商品成本50元，售价65元，卖出8件的总利润是多少元？", 120),
        ("水箱容量100升，已有35升，每分钟注入5升，几分钟注满？", 13),
    ]:
        add("word_math", prompt + "只输出最终数字，不要其他内容。", "number", expected)
    for prompt, expected in [
        ("甲比乙高，乙比丙高。最高的是谁？只输出甲、乙、丙之一。", "甲"),
        ("所有猫都是哺乳动物，小花是猫。能否确定小花是哺乳动物？只输出能或不能。", "能"),
        ("所有猫都怕水，小白怕水。能否确定小白是猫？只输出能或不能。", "不能"),
        ("A在B左边，C在B右边，三人排成一行。最右边是谁？只输出A、B、C之一。", "C"),
        ("袋中只有红球和蓝球，取出的球不是红球。它是什么颜色？只输出红色或蓝色。", "蓝色"),
        ("如果下雨，地面就湿。现在地面湿了。能否确定下过雨？只输出能或不能。", "不能"),
        ("每个工程师都会编程，小王是工程师。小王会编程吗？只输出会或不会。", "会"),
        ("箱子只装苹果和梨。已知没有苹果但箱子非空，里面有什么？只输出苹果或梨。", "梨"),
    ]:
        add("logic", prompt, "exact", expected)
    for values in ([9, 2, 7, 1], [12, -3, 0, 8], [4, 4, 2, 9], [31, 13, 22, 5]):
        add("format", f"将{list(values)}从小到大排序，只输出JSON数组，不要代码块。", "json", sorted(values))
    add("format", '从“姓名：张三，年龄：18”提取信息，只输出JSON对象，键为name和age，age必须为数字。不要代码块。', "json", {"name":"张三", "age":18})
    add("format", "只输出两个字段的JSON对象：ok为布尔值true，count为数字3。不要其他内容或代码块。", "json", {"ok":True, "count":3})
    add("format", "只输出大写英文单词YES，不要标点和解释。", "exact", "YES")
    add("format", "请严格输出两行，第一行为苹果，第二行为香蕉，不要其他内容。", "exact", "苹果\n香蕉")
    for prompt, expected in [("法国的首都是哪里？", "巴黎"), ("水的化学式是什么？", "H2O"), ("一个三角形有几条边？只输出数字。", "3"), ("一年有几个月？只输出数字。", "12")]:
        add("knowledge", prompt, "contains", expected)
    for prompt in ("你好，请用一句话介绍你能提供什么帮助。", "请用简短易懂的语言解释什么是过拟合，并举一个例子。", "请比较梯度下降和随机梯度下降的区别。", "写一封不超过100字的邮件，向导师说明实验推迟一天并致歉。", "小李说‘把学习率调得越大，训练一定越好’，这个说法是否正确？请简短解释。", "请将‘I enjoy learning machine learning.’翻译成自然的中文。"):
        add("open", prompt, "open")
    add("context", "我最喜欢的颜色是什么？只输出颜色。", "exact", "蓝色", history=[{"role":"user","content":"请记住，我最喜欢的颜色是蓝色。"},{"role":"assistant","content":"好的，你最喜欢蓝色。"}])
    add("context", "现在应当使用哪个编号？只输出编号。", "exact", "B27", history=[{"role":"user","content":"项目编号原来是A19，现在已经更新为B27，请使用新编号。"},{"role":"assistant","content":"明白，使用B27。"}])
    for prompt in ("你好。", "1加1等于几？直接回答即可。", "请解释什么是过拟合，不需要外部查询。"):
        add("tool_diagnostic", prompt, "no_tool", tools=TOOLS)
    for city in ("北京", "杭州"):
        add("tool_diagnostic", f"请查询{city}当前的天气。", "tool", {"name":"get_weather", "arguments":{"city":city}}, tools=TOOLS)
    add("tool_diagnostic", "请使用计算工具计算98765*4321-12345。", "tool", {"name":"calculate","arguments":{"expression":"98765*4321-12345"}}, tools=TOOLS)
    add("tool_diagnostic", "帮我查询今天的天气。", "clarify", tools=TOOLS)
    return cases


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


def run_generation(model, tokenizer, args, output):
    cases = build_cases()
    save(output/"cases.json", cases)  # 在实际生成之前落盘，后续不根据回答改题。
    path = output/"generations.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for index, case in enumerate(cases, 1):
            messages = case.get("history", []) + [{"role":"user", "content":case["prompt"]}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False, tools=case.get("tools"))
            result = {**case, **generate(model, tokenizer, ids, args.device, args.max_new_tokens)}
            result["passed"] = grade(case, result["raw_text"])
            stream.write(json.dumps(result, ensure_ascii=False)+"\n")
            stream.flush()
            print(f"case={index}/{len(cases)} id={case['id']} passed={result['passed']} tokens={result['generated_tokens']} stop={result['stop_reason']}", flush=True)


@torch.inference_mode()
def supervised_stats(logits, labels):
    # 与训练的移位和忽略规则相同；分块转FP32计算CE，避免151936词表的整段FP32副本。
    batch, positions = (labels[:, 1:] != -100).nonzero(as_tuple=True)
    if not len(batch):
        raise ValueError("没有有效监督目标")
    loss_sum, correct = 0.0, 0
    for start in range(0, len(batch), 64):
        b, p = batch[start:start+64], positions[start:start+64]
        scores = logits[b, p].float()
        targets = labels[b, p+1]
        loss = torch.nn.functional.cross_entropy(scores, targets, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("评测损失非有限")
        loss_sum += loss.item()
        correct += int((scores.argmax(-1) == targets).sum().item())
    return loss_sum, correct, len(batch)


@torch.inference_mode()
def run_numeric(model, tokenizer, args, output, metadata):
    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    signature = metadata["signature"]
    source_hash = sha256(args.data)
    if source_hash != signature["source_sha256"]:
        raise ValueError("评测语料 SHA256 与训练断点不一致")
    encoder = QwenSFTEncoder(tokenizer, signature["max_length"], signature["split_seed"], signature["val_fraction"])
    train, val = [], []
    for row in records:
        item = encoder(row)
        if any(x != -100 for x in item["labels"][1:]):
            (val if item.pop("validation") else train).append(item)
    print(f"重建划分：train={len(train)} val={len(val)}", flush=True)
    rng = random.Random(20261019)
    indices = sorted(rng.sample(range(len(train)), min(128, len(train))))
    save(output/"numeric_selection.json", {"train_indices":indices, "val":"all", "seed":20261019,
         "source_sha256":source_hash, "split_seed":signature["split_seed"], "max_length":signature["max_length"]})
    results = {}
    for name, rows in (("validation", val), ("train_subset", [train[i] for i in indices])):
        total_loss, total_tokens, correct = 0.0, 0, 0
        started = time.perf_counter()
        sync(args.device)
        for i, row in enumerate(rows, 1):
            tokens = torch.tensor([row["input_ids"]], device=args.device)
            labels = torch.tensor([row["labels"]], device=args.device)
            logits = model(tokens, use_cache=False).logits
            loss_sum, matched, count = supervised_stats(logits, labels)
            correct += matched
            total_loss += loss_sum
            total_tokens += count
            del logits, tokens, labels
            if i == 1 or i % 100 == 0:
                print(f"{name}={i}/{len(rows)} loss={total_loss/total_tokens:.6f}", flush=True)
        sync(args.device)
        mean = total_loss/total_tokens
        results[name] = dict(samples=len(rows), supervised_tokens=total_tokens, loss=mean,
                             ppl=math.exp(mean), token_accuracy=correct/total_tokens,
                             seconds=time.perf_counter()-started)
        save(output/"numeric.json", results)


@torch.inference_mode()
def run_benchmark(model, tokenizer, args, output):
    # 固定长度测吞吐；即使生成 EOS 仍继续 64 token，仅用于速度，不用于质量评分。
    source = tokenizer.encode("机器学习实验需要记录配置并验证结果。"*100, add_special_tokens=False)
    generate(model, tokenizer, source[:128], args.device, 8, stop_at_eos=False)
    records = []
    for length in (128, 512):
        for trial in range(1, 4):
            value = generate(model, tokenizer, source[:length], args.device, 64, stop_at_eos=False)
            value.pop("raw_text")
            value.pop("text")
            records.append({"trial":trial, **value})
            save(output/"benchmark.json", records)
            print(f"benchmark input={length} trial={trial} tok/s={value['tokens_per_second']:.2f}", flush=True)
    # 使用同一权重检查 KV cache 与完整前缀重算的最后位置 logits。
    ids = torch.tensor([source[:24]], device=args.device)
    full = model(ids, use_cache=False, logits_to_keep=1).logits.float()
    prefix = model(ids[:, :-1], use_cache=True, logits_to_keep=1)
    cached = model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True, logits_to_keep=1).logits.float()
    delta = (full-cached).abs()
    log_full = full.log_softmax(-1)
    log_cached = cached.log_softmax(-1)
    kl = (log_full.exp()*(log_full-log_cached)).sum(-1).mean().item()
    save(output/"cache_check.json", {"max_abs_diff":delta.max().item(), "mean_abs_diff":delta.mean().item(),
         "kl_full_to_cached":kl,
         "argmax_equal":bool(torch.equal(full.argmax(-1), cached.argmax(-1))),
         "finite":bool(torch.isfinite(full).all() and torch.isfinite(cached).all()),
         "note":"单个位置的BF16诊断，不是严格等价性证明。不同矩阵形状可能存在数值差异，需同时看分布KL；argmax相同不足以证明实现一致。"})


def report(args, output):
    metadata = json.loads((output/"metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output/"generations.jsonl").read_text(encoding="utf-8").splitlines()]
    numeric = json.loads((output/"numeric.json").read_text(encoding="utf-8"))
    benchmark = json.loads((output/"benchmark.json").read_text(encoding="utf-8"))
    cache = json.loads((output/"cache_check.json").read_text(encoding="utf-8"))
    scored = [r for r in rows if r["passed"] is not None and r["group"] != "tool_diagnostic"]
    groups = sorted({r["group"] for r in rows})
    val = numeric["validation"]
    lines = ["# Qwen3-1.7B Full SFT 本地评测 — 2026-10-19", "",
             f"固定规则主测试通过 **{sum(r['passed'] for r in scored)}/{len(scored)}**；工具诊断单列。完整验证 loss **{val['loss']:.6f}**，PPL **{val['ppl']:.4f}**。",
             "", "这是本项目固定诊断集，不是标准榜单；未确认题目与训练语料的近似重合。格式不符合评分规则也计为未通过。未加载原始 Base 权重作对照，不能量化 SFT 带来的提升。",
             "", "## 1. 权重与运行条件", "",
             f"- 权重：`{metadata['checkpoint']}`。",
             f"- checkpoint SHA256：`{metadata['checkpoint_sha256']}`。",
             f"- 训练进度：`{json.dumps(metadata['progress'], ensure_ascii=False)}`。",
             f"- 独立参数量：{metadata['unique_parameters']:,}；测试 dtype：{metadata['inference_dtype']}。",
             f"- GPU：{metadata['gpu']}；PyTorch：{metadata['torch']}；CUDA 构建：{metadata['cuda']}；Python：{metadata['python']}。",
             f"- 平台：{metadata['platform']}；CPU threads=2；时间：{metadata['time']}。",
             "- 使用同一官方提交的本地配置和 tokenizer，严格加载全部权重；忽略优化器状态。共享 embedding/lm_head 保持绑定。",
             "- 原始 checkpoint 参数为 FP32，本次为适配本地 6GB GPU 转为 BF16 推理；与服务器 FP32 参数＋BF16 autocast 的数值可能略有差异。",
             "- 问答使用 Qwen 聊天模板、enable_thinking=False、贪心解码、KV cache；没有强制工具调用，也没有执行真实外部工具。",
             f"- 每题独立上下文（context 组使用预先固定的对话）；max_new_tokens={args.max_new_tokens}。原始输出完整保存在本报告及 runs。",
             f"- 分词配置 SHA256：`{metadata['tokenizer_config_sha256']}`；模型配置 SHA256：`{metadata['config_sha256']}`。",
             "", "## 2. 固定题评分", "", "| 类别 | 通过/评分数 |", "| --- | --- |"]
    for group in groups:
        part = [r for r in rows if r["group"] == group and r["passed"] is not None]
        lines.append(f"| {group} | {sum(r['passed'] for r in part)}/{len(part)} |" if part else f"| {group} | 原始回答人工审阅，不自动打分 |")
    lines += ["", "评分规则：number 取最后非空行，要求只有数字或‘答案/结果：数字’；exact 要求完整匹配；JSON 使用原始 json.loads，不自动修复；工具要求工具名、参数值和结构匹配。no_tool 只检查是否避免调用，不代表文字内容必然正确。",
              "", "## 3. 损失与 token 指标", "", "| 划分 | 样本 | 有效监督 token | loss | PPL | token 准确率 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, value in numeric.items():
        lines.append(f"| {name} | {value['samples']} | {value['supervised_tokens']} | {value['loss']:.6f} | {value['ppl']:.4f} | {value['token_accuracy']:.2%} |")
    lines += ["", "按训练记录的 source SHA、split_seed、val_fraction 和 max_length 重建划分。验证集完整覆盖；训练子集固定种子抽取128条。仅 assistant 目标参与损失，结束标记参与，标签只移位一次。按有效 token 加权。PPL/token 准确率不代表推理正确率。",
              "", "## 4. 本机生成性能", "", "| 输入 token | 输出 token | 重复次数 | TTFT 中位数/秒 | 总吞吐中位数/token·s⁻¹ | 解码吞吐中位数/token·s⁻¹ | 峰值已分配显存最大值/MiB |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for length in (128,512):
        part = [r for r in benchmark if r['input_tokens']==length]
        lines.append(f"| {length} | 64 | {len(part)} | {statistics.median(r['first_token_seconds'] for r in part):.4f} | {statistics.median(r['tokens_per_second'] for r in part):.2f} | {statistics.median(r['decode_tokens_per_second'] for r in part):.2f} | {max(r.get('peak_allocated_mib',0) for r in part):.1f} |")
    lines += ["", "性能在预热后测量，batch=1；包含 Python 生成循环、逐步有限值检查、采样和 CUDA 同步，不包含 tokenizer 与模型加载；显存包含常驻权重，不包含驱动等全部进程占用。固定输出64 token，忽略 EOS，仅作性能测试。RTX3060 Laptop、Windows 和当前 SDPA 后端的结果不能当作4090 D服务器吞吐。",
              "", "## 5. 终止、重复和缓存诊断", "",
              f"- 终止原因：`{dict(Counter(r['stop_reason'] for r in rows))}`。",
              f"- 重复4-gram比例超过0.3的样本：{[r['id'] for r in rows if r['repeated_4gram_fraction']>0.3]}。",
              f"- KV cache 与重算：`{json.dumps(cache, ensure_ascii=False)}`。",
              "- 达到生成上限的回答可能不完整，不能将截断直接认定为模型不会作答。",
              "", "## 6. 失败案例与后续工作", "",
              f"- 主测试未通过题目：{[r['id'] for r in scored if not r['passed']]}。",
              "- 先区分内容错误、格式偏离与输出截断，再决定补充哪些 SFT 数据。",
              "- 本轮 Alpaca 不含结构化工具调用；工具项用于诊断现状，不用于宣称本轮训练达成了工具能力。",
              "- 下一轮候选：经检查的推理与工具混合数据，同时包含普通直接回答、无需工具、参数缺失与失败恢复样本。",
              "- 若比较 Full SFT/LoRA 效果，需要相同起点、训练数据和评测条件；若从本权重继续 LoRA，应标记为能力扩展实验。",
              "", "## 7. 全部原始回答", ""]
    for row in rows:
        lines += [f"### {row['id']} — {row['group']}", "", f"输入：{row['prompt']}", "",
                  f"期望/评分：`{json.dumps(row.get('expected'), ensure_ascii=False)}`；passed=`{row['passed']}`；stop=`{row['stop_reason']}`；生成 token={row['generated_tokens']}。", ""]
        if row.get("history"):
            lines += ["固定历史：", "```json", json.dumps(row["history"], ensure_ascii=False, indent=2), "```", ""]
        if row.get("tools"):
            lines += ["提供 get_weather(city) 和 calculate(expression) 工具描述；不执行工具。", ""]
        lines += ["模型原始输出：", "````text", row["raw_text"], "````", ""]
    lines += ["## 8. 产物与限制", "", f"- 原始 JSON/JSONL：`{output}`。",
              f"- 权重测试后 SHA256：`{metadata.get('checkpoint_sha256_after','尚未检查')}`。",
              "- 无标准榜单、外部模型裁判、真实工具服务或原始Base权重对照。本报告不宣称通用能力排名。", ""]
    review_path = output/"content_review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review["checkpoint_sha256"] != metadata["checkpoint_sha256"] or review["generations_sha256"] != sha256(output/"generations.jsonl"):
            raise ValueError("内容复核记录与当前权重或回答不匹配")
        failed_ids = {r["id"] for r in scored if not r["passed"]}
        if set(review["format_only"]) | set(review["content_error"]) != failed_ids:
            raise ValueError("内容复核未覆盖全部主测试失败题")
        summary = (f"逐题内容复核发现：上述{len(failed_ids)}条严格规则未通过题中，"
                   f"{len(review['format_only'])}条内容正确但格式不符，{len(review['content_error'])}条有内容错误。"
                   "该复核由本会话助手在生成后完成，非预先登记的自动成绩，也非独立人工标注；严格成绩保持不变。")
        lines.insert(3, summary)
        lines += ["## 9. 逐题内容复核与诊断", "", summary, "",
                  f"- 仅格式偏离：{review['format_only']}。",
                  f"- 内容错误：{review['content_error']}。", ""]
        for note in review["notes"]:
            lines.append("- "+note)
        lines.append("")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print("报告：", args.report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT/"runs/qwen-fullsft/checkpoint.pt"))
    parser.add_argument("--reference-dir", default=str(ROOT/"runs/qwen-base-reference"))
    parser.add_argument("--data", default=str(ROOT/"dataset/datasets/alpaca_gpt4_zh/alpaca_gpt4_data_zh.json"))
    parser.add_argument("--output-dir", default=str(ROOT/"runs/qwen-eval-20261019"))
    parser.add_argument("--report", default=str(ROOT/"test/test_log/full_sft/test_qwen_full_sft_2026-10-19.md"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--phase", choices=("all","generate","numeric","benchmark","report"), default="all")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "report":
        report(args, output)
        return
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dtype = torch.bfloat16 if torch.device(args.device).type == "cuda" else torch.float32
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("本评测 CUDA 模式需要 BF16 支持")
    print("计算 checkpoint SHA256（只读）...", flush=True)
    before = sha256(args.checkpoint)
    print("严格加载模型权重...", flush=True)
    model, metadata = load_model(args.checkpoint, args.reference_dir, args.device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.reference_dir, local_files_only=True, use_fast=True)
    metadata.update(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=before,
                    time=datetime.now().astimezone().isoformat(), python=sys.version.split()[0],
                    platform=platform.platform(), torch=str(torch.__version__), cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.device(args.device).type=="cuda" else "CPU",
                    inference_dtype=str(dtype), tokenizer_config_sha256=sha256(Path(args.reference_dir)/"tokenizer_config.json"),
                    config_sha256=sha256(Path(args.reference_dir)/"config.json"))
    save(output/"metadata.json", metadata)
    print(f"模型已加载，独立参数={metadata['unique_parameters']:,}；进度={metadata['progress']}", flush=True)
    if args.phase in ("all","generate"):
        run_generation(model, tokenizer, args, output)
    if args.phase in ("all","numeric"):
        run_numeric(model, tokenizer, args, output, metadata)
    if args.phase in ("all","benchmark"):
        run_benchmark(model, tokenizer, args, output)
    print("复核 checkpoint SHA256...", flush=True)
    metadata["checkpoint_sha256_after"] = sha256(args.checkpoint)
    if before != metadata["checkpoint_sha256_after"]:
        raise RuntimeError("测试前后权重文件发生变化")
    save(output/"metadata.json", metadata)
    if args.phase == "all":
        report(args, output)


if __name__ == "__main__":
    main()
