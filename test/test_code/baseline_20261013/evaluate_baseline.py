"""Reproducible local base-model evaluation; never updates checkpoint weights."""
import hashlib
import importlib.util
import json
import math
import platform
import random
import statistics
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
import tokenizers
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from dataset.dataset import TrainPilotTokenizer, PretrainDataset
from trainer.trainer import evaluate

spec = importlib.util.spec_from_file_location('base_entry', ROOT / 'test/test_code/base_model_test.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
OUT = ROOT / 'runs/base-eval-20261013'
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT = ROOT / 'runs/pre_train1-20261012/checkpoint.pt'
MANIFEST = ROOT / 'runs/pre_train1-20261012/run_manifest.json'
PROMPTS = [
    ('P01', '概念续写', '人工智能是一种'),
    ('P02', '解释指令', '介绍一下python。'),
    ('P03', '具体事实', '介绍一下电子科技大学。'),
    ('P04', '基础事实', '在标准大气压下，水的沸点是'),
    ('P05', '上下文保持', '小林把红色笔记本放进蓝色背包。他想拿出笔记本，应该先'),
    ('P06', '简单代码', '请用C++写一个函数 int add(int a, int b)，返回两个整数的和。'),
    ('P07', '故事续写', '清晨，小林走到公交站，突然发现钱包不见了，于是'),
    ('P08', '前缀信息保持', '我叫曹浩田。我的名字是'),
    ('P09', '事实解释', '天空看起来是蓝色的，因为'),
    ('P10', '英文续写', 'Machine learning is a'),
]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def sync():
    torch.cuda.synchronize()

def prepare_subset(manifest):
    """Reproduce full validation-file digest, reservoir-sample its records."""
    source = ROOT / 'dataset/pretrain_t2t_mini.jsonl'
    source_meta = json.loads((ROOT / 'dataset/SOURCE.json').read_text(encoding='utf-8'))
    raw_hash, val_hash = hashlib.sha256(), hashlib.sha256()
    rng = random.Random(20261013)
    sample = []
    val_count = 0
    threshold = int(0.01 * (1 << 64))
    started = time.perf_counter()
    with source.open('rb') as f, (OUT / 'validation_full.jsonl').open('wb') as val_file:
        for raw in f:
            raw_hash.update(raw)
            if not raw.strip():
                continue
            record = json.loads(raw)
            text = record['text']
            normalized = text.strip()
            if not normalized:
                continue
            bucket = hashlib.sha256(f'43\0{normalized}'.encode('utf-8')).digest()
            if int.from_bytes(bucket[:8], 'big') >= threshold:
                continue
            encoded = (json.dumps({'text': text}, ensure_ascii=False) + '\n').encode('utf-8')
            val_hash.update(encoded)
            val_file.write(encoded)
            val_count += 1
            if val_count <= 256:
                sample.append(encoded)
            else:
                j = rng.randrange(val_count)
                if j < 256:
                    sample[j] = encoded
    assert raw_hash.hexdigest() == source_meta['sha256'], 'raw corpus SHA mismatch'
    assert val_hash.hexdigest() == manifest['data_sha256']['val_path'], 'validation split SHA mismatch'
    sample_path = OUT / 'validation_subset.jsonl'
    sample_path.write_bytes(b''.join(sample))
    result = dict(raw_sha256=raw_hash.hexdigest(), full_val_sha256=val_hash.hexdigest(),
                  full_val_records=val_count, subset_records=len(sample), seed=20261013,
                  subset_sha256=sha256(sample_path), split_seed=43, val_fraction=0.01,
                  scan_seconds=time.perf_counter() - started)
    print('validation split verified:', json.dumps(result), flush=True)
    return sample_path, result

@torch.inference_mode()
def generation(model, tokenizer, prompt, mode):
    bos, eos, pad = [tokenizer.token_to_id(x) for x in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')]
    ids = [bos] + tokenizer.encode(prompt, add_special_tokens=False).ids
    tokens = torch.tensor([ids], dtype=torch.long, device='cuda')
    emitted = []
    stop = 'max_new_tokens'
    first = None
    forwards = 0
    torch.manual_seed(42)  # reset per case, independent of case order
    sync()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(96):
        if tokens.shape[1] >= model.config.max_seq_len:
            stop = 'context_limit'
            break
        scores = model(tokens)[0, -1, :].float().clone()
        forwards += 1
        assert torch.isfinite(scores).all().item()
        scores[pad] = scores[bos] = float('-inf')
        if mode == 'greedy':
            next_id = int(scores.argmax().item())
        else:
            scores /= 0.8
            cutoff = torch.topk(scores, 50).values[-1]
            probs = torch.softmax(scores.masked_fill(scores < cutoff, float('-inf')), dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        if first is None:
            sync()
            first = time.perf_counter() - started
        if next_id == eos:
            stop = 'eos'
            break
        emitted.append(next_id)
        tokens = torch.cat((tokens, torch.tensor([[next_id]], device='cuda', dtype=torch.long)), dim=1)
    sync()
    elapsed = time.perf_counter() - started
    # Numeric repetition diagnostic, not a human quality score.
    grams = [tuple(emitted[i:i+4]) for i in range(max(0, len(emitted)-3))]
    rep4 = 1 - len(set(grams)) / len(grams) if grams else None
    return dict(prompt_tokens=len(ids), generated_ids=emitted, generated_tokens=len(emitted),
                output=tokenizer.decode(emitted, skip_special_tokens=True), stop_reason=stop,
                forward_calls=forwards, first_token_seconds=first, elapsed_seconds=elapsed,
                output_tokens_per_second=len(emitted)/elapsed,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                repeated_token_4gram_fraction=rep4)

@torch.inference_mode()
def forward_bench(model):
    rows = []
    g = torch.Generator().manual_seed(42)
    for length in (16, 64, 256):
        x = torch.randint(3, model.config.vocab_size, (1, length), generator=g).to('cuda')
        for _ in range(5):
            y = model(x)
            del y
        sync()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(20):
            sync()
            t = time.perf_counter()
            y = model(x)
            sync()
            timings.append((time.perf_counter() - t)*1000)
            del y
        rows.append(dict(batch_size=1, input_tokens=length, warmups=5, repeats=20,
                         median_ms=statistics.median(timings), mean_ms=statistics.mean(timings),
                         min_ms=min(timings), max_ms=max(timings),
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
    return rows

def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('This recorded experiment requires CUDA')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert sha256(ROOT / 'dataset/tokenizer.json') == manifest['data_sha256']['tokenizer_path']
    before = sha256(CHECKPOINT)
    subset, split = prepare_subset(manifest)
    captured = []
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        model = entry.load_base_model(CHECKPOINT, torch.device('cuda'))
        tokenizer = Tokenizer.from_file(str(ROOT / 'dataset/tokenizer.json'))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        # Check instrumentation preserves the interactive entry's greedy output.
        expected = entry.generate_text(model, tokenizer, PROMPTS[0][2], torch.device('cuda'), 96, 0, 50)
        actual = generation(model, tokenizer, PROMPTS[0][2], 'greedy')['output']
        assert expected == actual, 'generation differs from existing entry'
        print('greedy instrumentation matches interactive entry', flush=True)
        enc = TrainPilotTokenizer(ROOT / 'dataset/tokenizer.json', model.config.max_seq_len)
        ds = PretrainDataset(subset, enc)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        sync()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        validation = evaluate(model, loader, 'cuda', amp_dtype=None)
        sync()
        validation.update(elapsed_seconds=time.perf_counter()-started,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                          dtype='float32', batch_size=8)
        validation['ppl'] = math.exp(validation['loss'])
        print('local validation:', json.dumps(validation), flush=True)
        full_validation = full_eval(model)
        bench = forward_bench(model)
        rows = []
        for mode in ('greedy', 'sample'):
            for pid, category, prompt in PROMPTS:
                record = dict(id=pid, category=category, mode=mode, prompt=prompt)
                record.update(generation(model, tokenizer, prompt, mode))
                rows.append(record)
                (OUT/'generation_results.partial.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f"{mode} {pid}: {record['generated_tokens']} tokens, {record['elapsed_seconds']:.3f}s, stop={record['stop_reason']}", flush=True)
        captured = sorted(set(str(w.message) for w in ws))
    after = sha256(CHECKPOINT)
    assert before == after, 'checkpoint changed'
    metrics = [json.loads(line) for line in (CHECKPOINT.parent/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    output = dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        checkpoint=str(CHECKPOINT), checkpoint_sha256=before, checkpoint_unchanged=True,
        checkpoint_bytes=CHECKPOINT.stat().st_size, model_config=vars(model.config),
        parameters=sum(p.numel() for p in model.parameters()),
        environment=dict(python=sys.version, torch=torch.__version__, tokenizers=tokenizers.__version__,
                         cuda_build=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                         gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                         platform=platform.platform(), threads=2, dtype='float32', kv_cache=False),
        manifest=manifest, historical_metrics=[r for r in metrics if r['event'] in ('epoch','initial_validation')],
        split=split, local_validation=validation, full_validation=full_validation, forward_benchmark=bench, generations=rows,
        settings=dict(seed_per_case=42, max_new_tokens=96, temperature_sample=0.8, top_k_sample=50,
                      prompt_format='BOS + plain text; no chat template', warmup='one full greedy generation plus equivalence check'),
        warnings=captured)
    (OUT/'results.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved', OUT/'results.json', flush=True)

def full_eval(model):
    enc = TrainPilotTokenizer(ROOT / 'dataset/tokenizer.json', model.config.max_seq_len)
    ds = PretrainDataset(OUT / 'validation_full.jsonl', enc)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    model.eval()
    sync()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss_sum, tokens = 0.0, 0
    from trainer.pretrain import next_token_loss
    with torch.inference_mode():
        for i, batch in enumerate(loader, 1):
            logits = model(batch['input_ids'].to('cuda'), attention_mask=batch['attention_mask'].to('cuda'))
            loss, count = next_token_loss(logits, batch['labels'].to('cuda'))
            assert torch.isfinite(loss).item()
            loss_sum += loss.item() * count
            tokens += count
            del logits, loss
            if i % 200 == 0:
                print(f'full validation {i}/{len(loader)}', flush=True)
    sync()
    result = dict(loss=loss_sum/tokens, ppl=math.exp(loss_sum/tokens), valid_tokens=tokens,
                  steps=len(loader), records=len(ds), batch_size=8, dtype='float32',
                  elapsed_seconds=time.perf_counter()-started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
    print('full validation:', json.dumps(result), flush=True)
    return result

if __name__ == '__main__':
    main()
