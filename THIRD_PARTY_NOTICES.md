# Third-party sources

TrainPilot is a project-owned PyTorch Dense pretraining implementation. Model layers, the Dataset, next-token loss and single-step training are implemented. Epoch training, validation and training-state checkpoints are still pending. Source study and rewriting retain the obligation to acknowledge reference work and preserve applicable notices.

## MiniMind

- Source: https://github.com/jingyaogong/minimind
- Pinned revision: f659b55761b754d306bd140573493a6543cafd7f
- Code license: Apache-2.0; retain its notices and license when adapting source.
- Studied files: model/model_minimind.py, dataset/lm_dataset.py, trainer/train_pretrain.py, trainer/trainer_utils.py.
- Scope: Dense, RMSNorm, RoPE, GQA, Q/K Norm and SwiGLU. Runtime uses this project's modules.
- Tokenizer: the 6400-token vocabulary is stored as assets/tokenizer.json and loaded via tokenizers. It was downloaded from model/tokenizer.json at the pinned revision through https://cdn.jsdelivr.net/gh/jingyaogong/minimind@f659b55761b754d306bd140573493a6543cafd7f/model/tokenizer.json.
- Tokenizer SHA256: 71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668.
- Original upstream LICENSE: assets/minimind.LICENSE, downloaded from the same pinned revision; SHA256 c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4.
- Code licensing does not establish the license of every dataset or weight file.

## Corpus and fixtures

The existing server experiment uses pretrain_t2t_mini.jsonl from jingyaogong/minimind_dataset at revision 312afb4f76391145c6902f765bb51691c09a12f5. Its card lists Apache-2.0 and CC-BY-NC-2.0 without a per-record mapping. Keep attribution and applicable restrictions. Raw corpus is outside this repository.

Source SHA256: 6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c.

The texts under tests/fixtures are self-authored pipeline checks, not a language-model benchmark. PyTorch and tokenizers retain their respective licenses. The project's own license has not been selected by its owner.
