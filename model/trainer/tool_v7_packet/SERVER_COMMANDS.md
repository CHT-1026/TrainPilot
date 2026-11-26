# V7 工具专项 LoRA SFT 服务器运行说明

2026-11-02。学校 RTX 4090 D，复用 `$HOME/.venvs/trainpilot-qwen`。
本包从**现有 V6 adapter 接续 SFT**，新建优化器；不是 GRPO，也不重新做全参数训练。
Full SFT底座仍为 `$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model`；本包已含冻结V6 adapter，不含约7GB底座。

## 1. 上传和解压

上传 `TrainPilot-Tool-SFT-V7-20261102.zip` 及同名 `.zip.sha256` 到网页终端的个人主目录。
请一次复制一个代码块。使用新目录，不覆盖旧项目。

```bash
cd "$HOME"
sha256sum -c TrainPilot-Tool-SFT-V7-20261102.zip.sha256
source "$HOME/.venvs/trainpilot-qwen/bin/activate"
python - <<'PY'
from pathlib import Path
import zipfile
target=Path.home()/'TrainPilot-Tool-SFT-V7'
target.mkdir(exist_ok=False)
with zipfile.ZipFile(Path.home()/'TrainPilot-Tool-SFT-V7-20261102.zip') as z:
    z.extractall(target)
print(target)
PY
cd "$HOME/TrainPilot-Tool-SFT-V7"
python verify_packet.py
```

若目录已存在，请先进入它检查是否已经解压或开始训练，不要重复覆盖。

## 2. 环境与底座

只补两个轻量依赖，复用已有PyTorch/Transformers/PEFT环境；不需要下载新模型或语料。

```bash
source "$HOME/.venvs/trainpilot-qwen/bin/activate"
cd "$HOME/TrainPilot-Tool-SFT-V7"
python -m pip install -r requirements-tools.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 3
export TRAINPILOT_SFT_MODEL="$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model"
nvidia-smi
ls "$TRAINPILOT_SFT_MODEL/config.json" "$TRAINPILOT_SFT_MODEL/model.safetensors.index.json"
```

若原环境已丢失，先建立环境并安装 `requirements.txt` 和 `requirements-tools.txt`，再继续。
不要将原始 Qwen Base 代替 Full SFT 导出底座，也不要以旧 V6 checkpoint.pt 恢复本轮。

## 3. 一次启动检查、短试跑、正式训练

```bash
cd "$HOME/TrainPilot-Tool-SFT-V7"
source "$HOME/.venvs/trainpilot-qwen/bin/activate"
export TRAINPILOT_SFT_MODEL="$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model"
mkdir -p runs/trainpilot_tool_lora_v7
nohup bash trainer/run_tool_sft_v7.sh all \
  > runs/trainpilot_tool_lora_v7/pipeline.log 2>&1 < /dev/null &
echo "后台进程 PID：$!"
```

先核验环境、包哈希、V6起点、资料轨迹与标签，再训练10次更新；随后自动恢复继续到完整一轮。
10次试跑包含在正式160次更新中。任何一步失败均停止，不会继续下一步。
网页终端断开不影响 `nohup` 后台进程。重复执行同一训练目录会被flock拒绝。

```bash
cd "$HOME/TrainPilot-Tool-SFT-V7"
tail -n 40 -f runs/trainpilot_tool_lora_v7/pipeline.log
```

Ctrl+C仅退出看日志。重新登录后先检查进程，不要直接再次启动：

```bash
pgrep -af '[r]un_tool_sft_v7.sh|[p]ython.*trainer.qwen_lora_sft'
tail -n 30 "$HOME/TrainPilot-Tool-SFT-V7/runs/trainpilot_tool_lora_v7/pipeline.log"
```

中断后恢复（会从本轮checkpoint继续，不重置优化器）：

```bash
cd "$HOME/TrainPilot-Tool-SFT-V7"
source "$HOME/.venvs/trainpilot-qwen/bin/activate"
export TRAINPILOT_SFT_MODEL="$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model"
nohup bash trainer/run_tool_sft_v7.sh train \
  >> runs/trainpilot_tool_lora_v7/pipeline.log 2>&1 < /dev/null &
```

## 4. 可选：训练前后同条件生成评测

完成训练后可分别加载V6/V7，测试同一批冻结的120条新工具环境题；不执行真实OA业务。
旧业务回放40条不在这个工具生成分母中，需后续独立业务回归。
两个命令串行运行，不能与训练同时占同一张卡。

```bash
cd "$HOME/TrainPilot-Tool-SFT-V7"
source "$HOME/.venvs/trainpilot-qwen/bin/activate"
export TRAINPILOT_SFT_MODEL="$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model"
nohup bash -c 'set -euo pipefail
bash trainer/run_tool_sft_v7.sh eval-before
bash trainer/run_tool_sft_v7.sh eval-after' \
  > runs/trainpilot_tool_lora_v7/evaluation.log 2>&1 < /dev/null &
```

结果分别为 `eval_before.jsonl`、`eval_after.jsonl` 及各自 `.summary.json`。
规则分数检查状态、必要事实、实际交付引用；不能称为独立语义准确率，也不能与同学RAG成绩直接比较。
裸JSON兼容调用和标准Hermes调用分别记录。评测文件已存在会拒绝覆盖。

## 5. 配置、数据与下载结果

- 训练1280条：960条完整工具轨迹＋320条原V6业务回放；验证160条，测试160条。
- 新工具轨迹包括单次检索、两跳补证、空结果改写、目录与分页读取、缺证、冲突、年份选择和澄清。
- 工具返回来自可执行的合成小型资料环境，尚不能代表真实企业语料泛化；未将同学1239题作为训练数据。
- 监督assistant调用和回答；system/user/tool内容屏蔽为-100；仅损失函数移位一次。
- 1 epoch，batch=1，累积8，有效batch=8；共160次更新，lr=1e-5，长度2048，BF16计算，梯度检查点。
- rank/alpha/目标层继承V6，不额外叠加一份LoRA。学习率按完整一轮调度，短试跑不会压缩调度。
- 预先做完整编码检查，不静默截掉最后回答；完整样例见 `dataset/datasets/trainpilot_agent/tool_v7/preview.md`。
- 4090 D 的本轮耗时尚未实测；先用试跑日志估算，不能把此前较短V6样本的耗时直接套用。

完成标志：日志 `completed_epochs=1; global_step=160`（默认配置），并生成以下文件：

```text
runs/trainpilot_tool_lora_v7/
  adapter/                 最终V7适配器与tokenizer
  epoch_01/adapter/        一轮导出
  checkpoint.pt           本轮优化器/RNG/进度恢复断点
  metrics.jsonl           训练与验证损失
  run_manifest.json       实际参数、数据SHA和起始adapter
  loss_curve.png          损失图
```

请下载整个 `runs/trainpilot_tool_lora_v7`，V6原权重与旧实验目录仍保留。
