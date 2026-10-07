"""学习练习：流式抽样 -> 样本内精确去重 -> 固定训练/验证划分。

只需填写 reservoir_sample 和 split_records。其余是读写与记录辅助代码。
使用标准库，在 CPU 上运行；无需加载模型或占用 GPU 显存。
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


EXPECTED_SHA256 = "6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c"
DATA_REVISION = "312afb4f76391145c6902f765bb51691c09a12f5"


def reservoir_sample(records, sample_size: int, rng: random.Random) -> list[dict]:
    """遍历所有有效记录，用蓄水池抽样保留至多 sample_size 条。"""
    if sample_size <= 0:
        raise ValueError("sample_size 必须为正")

    # TODO 1：实现算法并返回样本列表
    reservoir = []
    for i, record in enumerate(records):
        if i < sample_size:
            # 前 sample_size 条直接进池子
            reservoir.append(record)
        else:
            # 后续第 i 条，在 [0, i] 中均匀抽一个整数 j
            j = rng.randint(0, i)
            if j < sample_size:
                reservoir[j] = record
    return reservoir


def split_records(records: list[dict], val_fraction: float, rng: random.Random):
    """返回 (train_records, val_records)，输入已在样本内精确去重。"""
    if len(records) < 2 or not 0 < val_fraction < 1:
        raise ValueError("至少需要两条记录，且验证比例须在 0 与 1 之间")

    # 1. 复制列表，避免原地修改传入的 records
    shuffled = records.copy()

    # 2. 用传入的随机数生成器打乱副本
    rng.shuffle(shuffled)

    # 3. 计算验证条数：按比例取整，至少 1 条，且训练集至少留 1 条
    n_val = int(len(records) * val_fraction)
    if n_val < 1:
        n_val = 1
    elif n_val > len(records) - 1:
        n_val = len(records) - 1

    # 4. 切片：后 n_val 条作验证集，其余作训练集
    val_records = shuffled[-n_val:]
    train_records = shuffled[:-n_val]

    return train_records, val_records


def iter_valid_records(source: Path, stats: dict, source_digest):
    """逐行读 JSONL；规范化仅为 text.strip()，不合并内部空白。"""
    with source.open("rb") as handle:
        for line_no, raw in enumerate(handle, start=1):
            source_digest.update(raw)
            stats["source_lines"] += 1
            if not raw.strip():
                stats["blank_lines"] += 1
                continue
            try:
                item = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"第 {line_no} 行不是有效 UTF-8 JSON") from exc
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"第 {line_no} 行必须是包含字符串 text 的对象")
            text = item["text"].strip()
            if not text:
                stats["empty_texts"] += 1
                continue
            stats["valid_records"] += 1
            if stats["valid_records"] % 200_000 == 0:
                print(f"已扫描有效记录: {stats['valid_records']}", flush=True)
            yield {"text": text}
    stats["scan_complete"] = True


def deduplicate_sample(records):
    """只对抽取的样本精确去重，不声称清洗了整份源语料。"""
    seen = set()
    result = []
    for record in records:
        text = record["text"]
        if text not in seen:
            seen.add(text)
            result.append(record)
    return result


def check_split(original, train, val):
    train_texts = {item["text"] for item in train}
    val_texts = {item["text"] for item in val}
    assert train and val, "训练集与验证集均须非空"
    assert len(train_texts) == len(train), "训练集存在精确重复"
    assert len(val_texts) == len(val), "验证集存在精确重复"
    assert not train_texts & val_texts, "两侧存在精确重复文本"
    assert len(train) + len(val) == len(original), "划分丢失或增加了记录"
    assert train_texts | val_texts == {item["text"] for item in original}


def write_jsonl(path: Path, records):
    digest = hashlib.sha256()
    with path.open("xb") as handle:
        for record in records:
            raw = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            handle.write(raw)
            digest.update(raw)
    return digest.hexdigest()


def self_check():
    """检查边界、重复运行的一致性和划分守恒；不证明抽样分布正确。"""
    fixture = [{"text": f"fixture-{i}"} for i in range(100)]
    first = reservoir_sample(iter(fixture), 10, random.Random(42))
    again = reservoir_sample(iter(fixture), 10, random.Random(42))
    assert first == again, "相同 seed 应抽取相同样本"
    assert len(first) == 10
    assert len({item["text"] for item in first}) == 10
    assert all(item in fixture for item in first)
    assert reservoir_sample(iter(fixture[:3]), 10, random.Random(42)) == fixture[:3]
    assert reservoir_sample(iter([]), 10, random.Random(42)) == []
    original_copy = list(first)
    train, val = split_records(first, 0.1, random.Random(43))
    check_split(first, train, val)
    assert len(val) == 1
    assert first == original_copy, "不应修改传入列表"
    assert (train, val) == split_records(first, 0.1, random.Random(43))
    print("抽样结果:", [item["text"] for item in first])
    print("基础检查通过；抽样算法仍需结合实现审阅。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.source is None or args.output_dir is None:
        parser.error("需要 --source 和 --output-dir，或者使用 --self-check")
    if args.sample_size < 2 or not 0 < args.val_fraction < 1:
        parser.error("sample-size 至少为 2，val-fraction 须在 0 与 1 之间")

    output_paths = [args.output_dir / name for name in ("train.jsonl", "val.jsonl", "manifest.json")]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("输出文件已存在；请换一个 output-dir，保留已有划分")
    stats = dict(source_lines=0, blank_lines=0, empty_texts=0, valid_records=0, scan_complete=False)
    source_digest = hashlib.sha256()
    sampled = reservoir_sample(
        iter_valid_records(args.source, stats, source_digest), args.sample_size, random.Random(args.seed)
    )
    assert stats["scan_complete"], "必须扫描完整文件，不能只取前缀"
    assert source_digest.hexdigest() == EXPECTED_SHA256, "源文件 SHA256 与本次固定数据不符"
    assert len(sampled) == min(args.sample_size, stats["valid_records"]), "抽样条数不正确"
    unique = deduplicate_sample(sampled)
    train, val = split_records(unique, args.val_fraction, random.Random(args.seed + 1))
    check_split(unique, train, val)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_sha = write_jsonl(output_paths[0], train)
    val_sha = write_jsonl(output_paths[1], val)
    manifest = {
        "source": {"repository": "jingyaogong/minimind_dataset", "revision": DATA_REVISION,
                   "filename": args.source.name, "path": str(args.source.resolve()),
                   "bytes": args.source.stat().st_size, "sha256": source_digest.hexdigest(),
                   "card_license_tags": ["apache-2.0", "cc-by-nc-2.0"]},
        "preparation": {"sampling": "reservoir_over_valid_source_records",
                        "sample_size_requested": args.sample_size, "sampling_seed": args.seed,
                        "split_seed": args.seed + 1, "val_fraction": args.val_fraction,
                        "normalization": "strip_outer_whitespace",
                        "dedup_scope": "exact_text_within_sample_after_sampling",
                        "near_duplicate_checked": False, "tokenized_overlap_checked": False},
        "counts": {**stats, "sampled": len(sampled), "sample_duplicates_removed": len(sampled) - len(unique),
                   "unique_sample": len(unique), "train": len(train), "val": len(val)},
        "outputs": {"train": {"filename": "train.jsonl", "sha256": train_sha},
                    "val": {"filename": "val.jsonl", "sha256": val_sha}},
    }
    with output_paths[2].open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print("原文精确重复交集: 0；近似重复和 token 截断后重合尚未检查。")
    print("划分记录:", output_paths[2])
    print("数据准备完成；尚未进行真实语料训练。")


if __name__ == "__main__":
    main()
