"""Compare TrainPilot with the pinned upstream Dense MiniMind implementation.

Run in the optional reference environment containing transformers==4.57.6.
The formal model/training implementation does not depend on this environment.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from config.config import TrainPilotconfig
from model.model import TrainPilotForCausalLM
from trainer.pretrain import next_token_loss

UPSTREAM_SHA = 'f659b55761b754d306bd140573493a6543cafd7f'
UPSTREAM_FILE_SHA256 = '548f1203282dccb3be43227b11463f469b976afbab614df52828bdd246c3786f'


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def compare_tensors(actual, expected, *, rtol=1e-4, atol=1e-6):
    if actual.shape != expected.shape:
        raise AssertionError(f'Shape mismatch: {actual.shape} vs {expected.shape}')
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError('Non-finite result')
    difference = (actual.detach().float() - expected.detach().float()).abs()
    result = dict(shape=list(actual.shape), max_abs_diff=difference.max().item(),
                  mean_abs_diff=difference.mean().item(), rtol=rtol, atol=atol,
                  passed=torch.allclose(actual, expected, rtol=rtol, atol=atol))
    return result


def run_case(reference_module, *, flash_attn, padded):
    config = TrainPilotconfig(vocab_size=32, hidden_size=32, num_layers=2,
                             num_heads=4, num_kv_heads=2, intermediate_size=64,
                             max_seq_len=16, max_position_embeddings=16,
                             flash_attn=flash_attn, dropout=0.0)
    reference_config = reference_module.MiniMindConfig(
        vocab_size=config.vocab_size, hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers, num_attention_heads=config.num_heads,
        num_key_value_heads=config.num_kv_heads, intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings, flash_attn=flash_attn,
        dropout=0.0, rms_norm_eps=config.rms_norm_eps, rope_theta=config.rope_theta,
        tie_word_embeddings=True, initializer_range=config.initializer_range,
    )
    torch.manual_seed(42)
    own = TrainPilotForCausalLM(config).float().eval()
    reference = reference_module.MiniMindForCausalLM(reference_config).float().eval()
    # Initialization order is irrelevant: explicitly copy every parameter.
    reference.load_state_dict(own.state_dict(), strict=True)
    for name, value in own.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    inputs = torch.tensor([[1, 11, 12, 2, 9, 8, 2], [1, 7, 13, 3, 4, 8, 2]])
    mask = torch.ones_like(inputs)
    if padded:
        inputs[0, 4:] = 0
        mask[0, 4:] = 0
        inputs[1, 6:] = 0
        mask[1, 6:] = 0
    labels = inputs.clone()
    labels[mask == 0] = -100
    own_logits = own(inputs, attention_mask=mask)
    ref_outputs = reference(inputs, attention_mask=mask, labels=labels, use_cache=False)
    own_loss, valid_tokens = next_token_loss(own_logits, labels)
    with torch.no_grad():
        own_hidden, _ = own.model(inputs, attention_mask=mask)
    hidden = compare_tensors(own_hidden, ref_outputs.hidden_states)
    logits = compare_tensors(own_logits, ref_outputs.logits)
    loss = compare_tensors(own_loss, ref_outputs.loss)
    own_loss.backward()
    ref_outputs.loss.backward()
    own_parameters = dict(own.named_parameters())
    ref_parameters = dict(reference.named_parameters())
    if own_parameters.keys() != ref_parameters.keys():
        raise AssertionError('Named parameter sets differ')
    gradients = {}
    for name, parameter in own_parameters.items():
        if parameter.grad is None or ref_parameters[name].grad is None:
            raise AssertionError(f'Missing gradient: {name}')
        gradients[name] = compare_tensors(parameter.grad, ref_parameters[name].grad)
    result = dict(config=asdict(config), flash_attn_requested=flash_attn, padded=padded,
                  weights_copied_and_verified_exactly=True,
                  inputs=inputs.tolist(), attention_mask=mask.tolist(), labels=labels.tolist(),
                  valid_targets=valid_tokens, own_loss=own_loss.item(), reference_loss=ref_outputs.loss.item(),
                  hidden=hidden, logits=logits, loss=loss, gradients=gradients,
                  gradient_parameter_tensors=len(gradients),
                  max_gradient_abs_diff=max(x['max_abs_diff'] for x in gradients.values()))
    result['passed'] = all(x['passed'] for x in [hidden, logits, loss, *gradients.values()])
    # A deliberate error must be detected; avoid a vacuous comparison test.
    with torch.no_grad():
        reference.lm_head.weight[3, 0] += 0.25
        changed_logits = reference(inputs, attention_mask=mask, use_cache=False).logits
    negative_control = compare_tensors(own_logits, changed_logits)
    result['negative_control'] = dict(
        perturbation='reference.lm_head.weight[3,0] += 0.25 (tiny test model only)',
        max_abs_diff=negative_control['max_abs_diff'],
        error_detected=not negative_control['passed'],
    )
    result['passed'] = result['passed'] and result['negative_control']['error_detected']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-file', default='runs/minimind-reference-f659b55/model_minimind.py')
    parser.add_argument('--output', default='runs/minimind-parity/results.json')
    args = parser.parse_args()
    reference_path = resolve_path(args.reference_file).resolve()
    if not reference_path.is_file():
        parser.error('Download the pinned upstream model/model_minimind.py first')
    source_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if source_digest != UPSTREAM_FILE_SHA256:
        parser.error('Reference source SHA256 differs from the pinned file')
    import transformers
    if transformers.__version__ != '4.57.6':
        parser.error('Run in the reference environment with transformers==4.57.6')
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('trainpilot_minimind_reference', reference_path)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    cases = []
    for flash_attn in (False, True):
        for padded in (False, True):
            result = run_case(reference_module, flash_attn=flash_attn, padded=padded)
            cases.append(result)
            print(f"flash_requested={flash_attn} padded={padded}: pass={result['passed']} "
                  f"logits_diff={result['logits']['max_abs_diff']:.3e} "
                  f"loss_diff={result['loss']['max_abs_diff']:.3e} "
                  f"gradient_diff={result['max_gradient_abs_diff']:.3e}", flush=True)
    record = dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  upstream_commit=UPSTREAM_SHA,
                  upstream_url=f'https://raw.githubusercontent.com/jingyaogong/minimind/{UPSTREAM_SHA}/model/model_minimind.py',
                  reference_file=str(reference_path),
                  reference_file_sha256=source_digest,
                  python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
                  device='cpu', dtype='float32', kv_cache=False,
                  passed=all(x['passed'] for x in cases), cases=cases)
    destination = resolve_path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved:', destination)
    if not record['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
