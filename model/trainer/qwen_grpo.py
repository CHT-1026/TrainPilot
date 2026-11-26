"""单卡原生 PyTorch GRPO 试验：V5 LoRA 策略 + 同底座的冻结 V5 reference。

仅处理可精确核验的业务意图/字段。每题串行采样 G 个完整回答；不使用价值模型。
采样使用 temperature=1/top_p=1/top_k=0，使 old logprob 与实际采样分布一致。
复用同一冻结底座的两份 adapter 节省显存；reference 不是关闭 adapter 的底座。
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
import torch.nn.functional as F

from model.trainer.agent_rewards import intent_reward


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_advantages(rewards, eps=1e-4):
    r = torch.as_tensor(rewards, dtype=torch.float32)
    if r.numel() < 2 or not torch.isfinite(r).all():
        raise ValueError("GRPO 每组至少两个有限奖励")
    return (r - r.mean()) / (r.std(unbiased=False) + eps)


def grpo_loss(logp, old_logp, ref_logp, advantage, beta=.02, clip=.2):
    """一条回答、仅 completion token 的原始 GRPO 序列均值目标。"""
    old_logp, ref_logp = old_logp.detach(), ref_logp.detach()
    ratio = (logp - old_logp).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    delta = ref_logp - logp
    kl = delta.exp() - delta - 1  # sampled per-token k3 estimator
    loss = (-surrogate + beta * kl).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("GRPO loss 非有限，请检查策略漂移与学习率")
    return loss, {"kl": kl.detach().mean().item(),
                  "clip_fraction": ((ratio.detach() - 1).abs() > clip).float().mean().item()}


def activate(model, name="default", training=False):
    model.set_adapter(name)
    # set_adapter 会切换 requires_grad；明确只允许策略 default 的 LoRA 更新。
    for key, p in model.named_parameters():
        p.requires_grad_(name == "default" and "lora_" in key and ".default." in key)
    model.train(training)


def completion_logps(model, prompt, completion, device):
    tokens = torch.tensor([prompt + completion], device=device)
    n = len(completion)
    # 只投影最后 n+1 个隐藏状态，避免为整段长 prompt 分配大词表 logits。
    logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                   use_cache=False, logits_to_keep=n + 1).logits[:, :-1].float()
    targets = torch.tensor([completion], device=device)
    return F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1))[0, :, 0]


def balanced_order(rows, seed):
    """每轮每条样本一次；前缀轮转不同场景，避免短试跑只看到请假题。"""
    rng, buckets = random.Random(seed), defaultdict(list)
    for i, row in enumerate(rows):
        buckets[row["category"]].append(i)
    keys = list(buckets)
    rng.shuffle(keys)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    order = []
    while keys:
        for key in list(keys):
            order.append(buckets[key].pop())
            if not buckets[key]:
                keys.remove(key)
    return order


def encode_rows(rows, tokenizer, limit):
    encoded = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=False)
        if len(ids) > limit:
            raise ValueError(f"{row['id']} prompt={len(ids)} 超过 {limit}；不要截掉业务上下文")
        intent_reward(json.dumps(row["gold"], ensure_ascii=False), row["gold"])
        encoded.append({**row, "input_ids": ids})
    return encoded


def load_policy(args):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    adapter_tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    if tokenizer.get_vocab() != adapter_tokenizer.get_vocab() or tokenizer.chat_template != adapter_tokenizer.chat_template:
        raise ValueError("Full SFT 底座与 V5 adapter 的 tokenizer/模板不一致")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                               torch_dtype=dtype, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True, autocast_adapter_dtype=True)
    model.load_adapter(args.adapter, adapter_name="reference", is_trainable=False, autocast_adapter_dtype=True)
    # 采样、old/reference logprob、训练 logprob 必须使用同一种关闭 dropout 的策略。
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(args.device)
    activate(model)
    return model, tokenizer


def select_probe_pool(rows, records, probe_id):
    """只选本轮训练侧probe的混合组输入；训练时重新采样，不复用probe回答。"""
    by_id = {r["id"]: r for r in rows}
    selected = set()
    for record in records:
        if record["probe_id"] != probe_id:
            continue
        row = by_id[record["id"]]
        rewards = {intent_reward(a["text"], row["gold"])["reward"] if a["ended"] else 0.
                   for a in record["answers"]}
        if len(rewards) > 1:
            selected.add(row["id"])
    return [r for r in rows if r["id"] in selected]


def eos_ids(model, tokenizer):
    ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    configured = model.config.eos_token_id
    ids.update(configured if isinstance(configured, list) else [configured])
    return sorted(i for i in ids if i is not None)


def generate_one(model, tokenizer, row, args, sample):
    from transformers import GenerationConfig
    activate(model)
    prompt = torch.tensor([row["input_ids"]], device=args.device)
    generation = GenerationConfig(
        do_sample=sample, **({"temperature": 1., "top_p": 1., "top_k": 0} if sample else {}), repetition_penalty=1.,
        max_new_tokens=args.max_completion_tokens, eos_token_id=eos_ids(model, tokenizer),
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, use_cache=True,
    )
    with torch.no_grad(), autocast(args):
        result = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                generation_config=generation)[0, prompt.shape[1]:].tolist()
    ended = bool(result) and result[-1] in eos_ids(model, tokenizer)
    text = tokenizer.decode(result, skip_special_tokens=True)
    score = intent_reward(text, row["gold"])
    if not ended:
        score["reward"] = 0.  # 长度上限不当作正常完成，明确记入截断计数。
    return {"tokens": result, "text": text, "raw_text": tokenizer.decode(result, skip_special_tokens=False),
            "ended": ended, **score}


def autocast(args):
    return torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else nullcontext()


def rollout(model, tokenizer, row, args):
    answers = [generate_one(model, tokenizer, row, args, True) for _ in range(args.group_size)]
    activate(model)
    with torch.no_grad(), autocast(args):
        old = [completion_logps(model, row["input_ids"], a["tokens"], args.device).cpu() for a in answers]
    activate(model, "reference")
    with torch.no_grad(), autocast(args):
        ref = [completion_logps(model, row["input_ids"], a["tokens"], args.device).cpu() for a in answers]
    activate(model)
    return answers, old, ref


def evaluate(model, tokenizer, rows, args, output, label):
    selected = balanced_order(rows, args.seed + 1000)
    if args.eval_limit:
        selected = selected[:args.eval_limit]
    records = [{"id": rows[i]["id"], "category": rows[i]["category"],
                **generate_one(model, tokenizer, rows[i], args, False)} for i in selected]
    write_json(output / f"{label}_answers.json", records)
    result = {"label": label, "total": len(records), "exact": sum(r["reward"] for r in records),
              "truncated": sum(not r["ended"] for r in records), "case_ids": [r["id"] for r in records]}
    write_json(output / f"{label}.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def fingerprint(args):
    base = Path(args.model).resolve()
    return {"model": str(base), "base_config_sha256": digest(base / "model.config.json"),
            "base_weight_files": {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                                  for p in sorted(base.glob("*.safetensors"))},
            "initial_adapter_sha256": digest(Path(args.adapter) / "adapter_model.safetensors"),
            "initial_adapter_config_sha256": digest(Path(args.adapter) / "adapter_config.json"),
            "data_sha256": {s: digest(Path(args.data_dir) / f"{s}.jsonl") for s in ("train", "val", "test")},
            **{k: getattr(args, k) for k in ("group_size", "max_prompt_tokens", "max_completion_tokens",
                "updates_per_group", "learning_rate", "beta", "clip", "seed", "device", "train_on_probed_mixed")},
            "sampling": "temperature=1,top_p=1,top_k=0,repetition_penalty=1", "dropout": 0,
            "trainer_code_sha256": digest(__file__),
            "reward_version": digest(Path(__file__).with_name("agent_rewards.py"))}


def save_checkpoint(model, tokenizer, optimizer, output, signature, progress):
    from peft import get_peft_model_state_dict
    activate(model)
    state = {"signature": signature, "progress": progress,
             "weights": {k: v.cpu() for k, v in get_peft_model_state_dict(model).items()},
             "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = output / "checkpoint.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(output / "checkpoint.pt")
    model.save_pretrained(output / "adapter", selected_adapters=["default"])
    tokenizer.save_pretrained(output / "adapter")


def train(model, tokenizer, rows, val, args, output, signature):
    from peft import set_peft_model_state_dict
    activate(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0., foreach=False)
    progress = {"global_step": 0, "groups": 0, "mixed_groups": 0, "cycle": 0, "cursor": 0}
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("断点与当前底座、初始V5、数据或GRPO参数不一致")
        set_peft_model_state_dict(model, state["weights"], adapter_name="default")
        optimizer.load_state_dict(state["optimizer"])
        progress = state["progress"]
        torch.set_rng_state(state["torch_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        before = json.loads((output / "before_val.json").read_text(encoding="utf-8"))
        selected = balanced_order(val, args.seed + 1000)
        if args.eval_limit:
            selected = selected[:args.eval_limit]
        if before["case_ids"] != [val[i]["id"] for i in selected]:
            raise ValueError("恢复训练请保持相同的验证子集，不能更改 eval-limit")
        print("恢复GRPO:", progress, flush=True)
    elif (output / "checkpoint.pt").exists():
        raise FileExistsError("结果目录已有断点，请 --resume 或使用新目录")
    if progress["global_step"] == 0:
        evaluate(model, tokenizer, val, args, output, "before_val")
    started = time.monotonic()
    while progress["global_step"] < args.max_updates:
        order = balanced_order(rows, args.seed + progress["cycle"])
        if progress["cursor"] == len(order):
            progress["cycle"] += 1
            progress["cursor"] = 0
            continue
        row = rows[order[progress["cursor"]]]
        answers, old, ref = rollout(model, tokenizer, row, args)
        rewards = [a["reward"] for a in answers]
        advantages = group_advantages(rewards)
        append_json(output / "rollouts.jsonl", {"group": progress["groups"], "id": row["id"],
                    "category": row["category"], "answers": answers, "advantages": advantages.tolist()})
        for inner in range(args.updates_per_group):
            activate(model, training=True)
            optimizer.zero_grad(set_to_none=True)
            total_loss, kl, clipped = 0., 0., 0.
            for i, answer in enumerate(answers):
                with autocast(args):
                    logp = completion_logps(model, row["input_ids"], answer["tokens"], args.device)
                    loss, stats = grpo_loss(logp, old[i].to(args.device), ref[i].to(args.device),
                                            advantages[i].to(args.device), args.beta, args.clip)
                    (loss / args.group_size).backward()
                total_loss += loss.detach().item() / args.group_size
                kl += stats["kl"] / args.group_size
                clipped += stats["clip_fraction"] / args.group_size
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True).item()
            optimizer.step()
            progress["global_step"] += 1
            record = {"step": progress["global_step"], "group": progress["groups"], "inner": inner,
                      "reward_mean": sum(rewards) / len(rewards), "mixed_group": len(set(rewards)) > 1,
                      "loss": total_loss, "sampled_kl": kl, "clip_fraction": clipped, "grad_norm": norm,
                      "truncated": sum(not a["ended"] for a in answers)}
            append_json(output / "metrics.jsonl", record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        progress["cursor"] += 1
        progress["groups"] += 1
        progress["mixed_groups"] += int(len(set(rewards)) > 1)
        if progress["global_step"] % args.save_steps == 0:
            save_checkpoint(model, tokenizer, optimizer, output, signature, progress)
    save_checkpoint(model, tokenizer, optimizer, output, signature, progress)
    evaluate(model, tokenizer, val, args, output, "after_val")
    write_json(output / "completed.json", {**progress, "has_task_signal": bool(progress["mixed_groups"]),
               "seconds": time.monotonic() - started,
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None})
    print("GRPO_PILOT_COMPLETE", {**progress, "output": str(output),
                                  "has_task_signal": bool(progress["mixed_groups"])}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON默认配置，命令行可覆盖")
    parser.add_argument("mode", choices=("check", "probe", "train", "eval"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="runs/init_v5/adapter")
    parser.add_argument("--data-dir", default="model/dataset/datasets/trainpilot_agent/grpo_v1")
    parser.add_argument("--output-dir", default="runs/trainpilot_grpo_pilot")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1792)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--updates-per-group", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=.02)
    parser.add_argument("--clip", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-prompts", type=int, default=16)
    parser.add_argument("--probe-file", help="训练前通过的采样检查摘要")
    parser.add_argument("--eval-limit", type=int, default=32, help="0=整个划分；默认固定分层前32题")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--resume")
    parser.add_argument("--policy-adapter", help="eval加载GRPO输出adapter；reference仍为初始V5")
    parser.add_argument("--train-on-probed-mixed", action="store_true", help="仅训练本轮probe确认有奖励差异的训练候选")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        parser.set_defaults(**json.loads(preliminary.config.read_text(encoding="utf-8")))
    args = parser.parse_args()
    if (args.group_size < 2 or args.updates_per_group < 1 or args.max_updates < 1
            or args.save_steps < 1 or args.probe_prompts < 1 or args.eval_limit < 0
            or args.max_updates % args.updates_per_group or args.save_steps % args.updates_per_group):
        parser.error("每组至少2条；更新上限和保存间隔必须为组内更新次数的正整数倍")
    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA/BF16不可用；先检查环境")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(args)
    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text(encoding="utf-8"))
    for split, info in manifest["splits"].items():
        if signature["data_sha256"][split] != info["sha256"]:
            raise ValueError(f"{split} 数据与包内清单不一致")
    probe = None
    if args.mode == "train":
        probe = json.loads(Path(args.probe_file or output / "probe_summary.json").read_text(encoding="utf-8"))
        if probe["signature"] != signature or not probe["mixed_groups"]:
            raise ValueError("采样检查配置不同或没有混合奖励组；本次不加载V5，请先审阅候选")
    model, tokenizer = load_policy(args)
    data = {s: encode_rows(read_rows(Path(args.data_dir) / f"{s}.jsonl"), tokenizer, args.max_prompt_tokens)
            for s in ("train", "val", "test")}
    write_json(output / "run_config.json", {"signature": signature,
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "counts": {s: len(r) for s, r in data.items()}, "loss": "original GRPO sequence-mean clipped + k3 KL",
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
               "reference": "second frozen adapter copied from V5; shared frozen Full SFT base"})
    if args.mode == "check":
        row = data["train"][0]
        target = tokenizer.encode(json.dumps(row["gold"], ensure_ascii=False), add_special_tokens=False)
        with torch.no_grad(), autocast(args):
            initial = completion_logps(model, row["input_ids"], target, args.device)
            activate(model, "reference")
            reference = completion_logps(model, row["input_ids"], target, args.device)
        diff = (initial - reference).abs().max().item()
        if diff > 1e-5:
            raise ValueError(f"初始策略与V5 reference不一致：{diff}")
        activate(model)
        print("CHECK_OK", {"initial_reference_max_logp_diff": diff, "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}, flush=True)
    elif args.mode == "probe":
        result = []
        probe_id = str(time.time_ns())
        for i in balanced_order(data["train"], args.seed)[:args.probe_prompts]:
            row = data["train"][i]
            answers = [generate_one(model, tokenizer, row, args, True) for _ in range(args.group_size)]
            rewards = [a["reward"] for a in answers]
            record = {"probe_id": probe_id, "id": row["id"], "category": row["category"], "answers": answers,
                      "rewards": rewards, "mixed": len(set(rewards)) > 1}
            append_json(output / "probe_answers.jsonl", record)
            result.append(record)
            print("probe", len(result), rewards, flush=True)
        summary = {"probe_id": probe_id, "signature": signature, "groups": len(result), "mixed_groups": sum(r["mixed"] for r in result),
                   "mean_reward": sum(sum(r["rewards"]) for r in result) / (len(result) * args.group_size),
                   "truncated": sum(not a["ended"] for r in result for a in r["answers"])}
        write_json(output / "probe_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    elif args.mode == "train":
        pool = data["train"]
        if args.train_on_probed_mixed:
            records = read_rows(Path(args.probe_file or output / "probe_summary.json").with_name("probe_answers.jsonl"))
            pool = select_probe_pool(pool, records, probe["probe_id"])
            if not pool: raise ValueError("重新核验后没有有效训练候选")
            signature = {**signature, "train_pool_ids": [r["id"] for r in pool]}
            write_json(output / "train_pool.json", {"probe_id": probe["probe_id"], "candidate_count": len(data["train"]),
                       "effective_count": len(pool), "ids": signature["train_pool_ids"], "resample_during_training": True})
            run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            run_config.update(signature=signature, effective_train_count=len(pool))
            write_json(output / "run_config.json", run_config)
            print("训练有效候选:", len(pool), "/", len(data["train"]), flush=True)
        train(model, tokenizer, pool, data["val"], args, output, signature)
    else:
        if args.policy_adapter:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(str(Path(args.policy_adapter) / "adapter_model.safetensors")), adapter_name="default")
        evaluate(model, tokenizer, data[args.eval_split], args, output, "eval_" + args.eval_split)


if __name__ == "__main__":
    main()
