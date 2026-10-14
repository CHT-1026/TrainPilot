# Third-party sources

TrainPilot is a project-owned PyTorch Dense pretraining implementation. It supports float32 CPU checks and single-CUDA-device BF16 pretraining, token-weighted gradient accumulation, gradient clipping, cosine learning-rate decay, validation and checkpoints after parameter updates. Source study and rewriting retain the obligation to acknowledge reference work and preserve applicable notices.

## MiniMind

- Source: https://github.com/jingyaogong/minimind
- Pinned revision: f659b55761b754d306bd140573493a6543cafd7f
- Code license: Apache-2.0; retain its notices and license when adapting source.
- Studied files: model/model_minimind.py, dataset/lm_dataset.py, trainer/train_pretrain.py, trainer/trainer_utils.py.
- Scope: Dense, RMSNorm, RoPE, GQA, Q/K Norm and SwiGLU. Runtime uses this project's modules.
- The pretrain configuration targets the pinned MiniMind-3 Dense architecture: hidden_size=768, layers=8, Q heads=8, KV heads=4, FFN width=2432, vocabulary=6400. This is approximately 64M parameters; it is a reproduction target, not a claim of matching the upstream released model's quality.
- Initialization is implemented explicitly with normal-distribution standard deviation 0.02, matching the default initialization used through MiniMind's post_init and transformers 4.57.6. Model computation remains native PyTorch. The SFT data layer uses datasets 3.6.0 and transformers 4.57.6 for loading and chat tokenization.
- Training differences: a separate validation split is retained, and gradient accumulation is weighted by valid supervised tokens rather than a simple average of micro-batch means. Batch size and speed still require GPU measurement.
- Tokenizer: the 6400-token vocabulary is stored as dataset/tokenizer.json and loaded via tokenizers. It was downloaded from model/tokenizer.json at the pinned revision through https://cdn.jsdelivr.net/gh/jingyaogong/minimind@f659b55761b754d306bd140573493a6543cafd7f/model/tokenizer.json.
- Tokenizer SHA256: 71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668.
- SFT tokenizer configuration and chat template: dataset/tokenizer_config.json, copied from model/tokenizer_config.json at the same pinned revision; SHA256 d7cd6a60c9f191c4f9ffec69b2eb289ad2f960366724f328b681ae1e46e4c110. AutoTokenizer can load it and the vocabulary locally without downloading model weights.
- Original upstream LICENSE: model/minimind.LICENSE, downloaded from the same pinned revision; SHA256 c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4.
- Code licensing does not establish the license of every dataset or weight file.

## Corpus and fixtures

The pretraining corpus is pretrain_t2t_mini.jsonl from jingyaogong/minimind_dataset at revision 312afb4f76391145c6902f765bb51691c09a12f5. Its card lists Apache-2.0 and CC-BY-NC-2.0 without a per-record mapping. Keep attribution and applicable restrictions. A copy is stored at dataset/datasets/pretrain_t2t_mini.jsonl through Git LFS; dataset/SOURCE.json records its source, checksum and license tags.

Source SHA256: 6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c.

The SFT corpus is sft_t2t_mini.jsonl from the same dataset repository and pinned revision. It is downloaded locally to dataset/datasets/sft_t2t_mini.jsonl and excluded from Git; dataset/SFT_SOURCE.json records its source, upstream checksum, license tags and completed structural validation. Downloading does not establish permission beyond the applicable upstream licenses.

SFT source SHA256: abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518.

The texts under dataset/datasets/fixtures are self-authored pipeline checks, not a language-model benchmark. PyTorch and tokenizers retain their respective licenses. The project's own license has not been selected by its owner.
