"""Render the recorded experiment as Markdown without modifying baseline files."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/base-eval-20261013'
RAW = ROOT / 'test_log/base_model_20261013'
TARGET = ROOT / 'test_log/test_base_model_10.13,md'
r = json.loads(((OUT/'results.json') if (OUT/'results.json').is_file() else (RAW/'results.json')).read_text(encoding='utf-8'))
c = json.loads(((OUT/'code_results.json') if (OUT/'code_results.json').is_file() else (RAW/'code_results.json')).read_text(encoding='utf-8'))
full = r['full_validation']
assert full['valid_tokens'] == r['historical_metrics'][-1]['validation']['valid_tokens']
assert r['checkpoint_unchanged']
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])
def mode_summary(mode):
    xs = [x for x in r['generations'] if x['mode']==mode]
    grams = sum(max(x['generated_tokens']-3,0) for x in xs)
    duplicates = sum((x['repeated_token_4gram_fraction'] or 0)*max(x['generated_tokens']-3,0) for x in xs)
    return [mode, len(xs), sum(x['generated_tokens'] for x in xs),
            f"{sum(x['generated_tokens'] for x in xs)/sum(x['elapsed_seconds'] for x in xs):.2f}",
            f"{statistics.median(x['first_token_seconds']*1000 for x in xs):.2f}",
            f"{max(x['peak_allocated_mib'] for x in xs):.2f}",
            sum(x['stop_reason']=='eos' for x in xs),
            sum(x['stop_reason']=='max_new_tokens' for x in xs), f'{duplicates/grams:.2%}']

parts = [f'''# TrainPilot 基础模型测试日志 — 2026-10-13

## 1. 实验信息与结论

- 实验编号：BASE-20261013-01。
- 模型阶段：完成两轮预训练的 Dense 基础语言模型；本次没有执行 SFT 或更新参数。
- 生成实验记录时间：{r['timestamp']}（北京时间）。完整验证复测在同一晚随后完成。
- 权重基线：`runs/pre_train1-20261012/checkpoint.pt`。
- 测试对象：63,912,192 个独立参数，约 64M。
- 当前代码 HEAD：`{revision}`。工作树含本地修改；HEAD 本身不能唯一标识本次运行的所有文件，具体代码 SHA256 见第 9 节。训练日志未记录训练时 Git SHA，不能据此证明训练与当前源码逐字一致。
- 记录来源：服务器保存的 `metrics.jsonl` / `run_manifest.json`，以及今天实际运行的本地实验。历史人工对话仅用于选择测试题，下面的输出全部为本次重新生成。

**结论：模型已学习到部分语言和概念表达，但当前的事实可靠性、上下文保持和生成稳定性不足。** 完整验证复测 loss={full['loss']:.6f}、PPL={full['ppl']:.4f}；这些指标不代表问答准确率。概念续写较流畅，但固定用例暴露了大学信息编造、姓名保持失败及重复循环。简单 C++ 加法函数在贪心生成下编译并通过 7 个断言；采样生成引入未声明变量，编译失败。

本次没有运行 MiniMind 对照模型，没有测得与其质量差距或数值一致性；不能宣称已达到或落后于某个官方权重的定量水平。

## 2. 模型与训练基线

''' + table(['项目','配置'], [
    ['模型结构','Dense / Pre-Norm / RMSNorm / RoPE / GQA / Q-K Norm / SwiGLU / 共享输入输出嵌入'],
    ['词表 / 层数 / 隐藏维度','6400 / 8 / 768'],
    ['Q 头 / KV 头 / 每头维度','8 / 4 / 96'],
    ['FFN 中间维度','2432'],
    ['训练截断长度 / RoPE buffer 上限','340 / 32768；buffer 上限不代表模型已学会长上下文'],
    ['训练精度 / epoch','BF16 autocast / 2'],
    ['优化器','AdamW；lr=0.0005，weight_decay=0.01'],
    ['梯度累积 / 裁剪','batch_size=32，accumulation_steps=8，有效 batch 约 256，clip=1.0'],
    ['调度 / 训练种子','余弦衰减至初始 LR 的 0.1 / seed=42'],
    ['完成的优化器更新数','9826'],
    ['每轮有效训练目标 token','262,273,758；两轮累计 524,547,516，包含重复遍历'],
    ['训练服务器软件','Python 3.12.3 / PyTorch 2.8.0+cu128 / CUDA 构建 12.8'],
    ['训练 GPU','RTX 4090，来自此前服务器环境记录；run_manifest 本身没有保存 GPU 型号'],
    ['checkpoint 文件大小',f"{r['checkpoint_bytes']:,} bytes / {r['checkpoint_bytes']/1024**2:.2f} MiB；包含优化器等状态，不是纯权重大小"],
]) + '''

## 3. 数据来源、划分与校验

原始语料为 `jingyaogong/minimind_dataset` 的 `pretrain_t2t_mini.jsonl`，固定 revision `312afb4f76391145c6902f765bb51691c09a12f5`。本地原文件的 SHA256 与 `dataset/SOURCE.json` 一致。

本次按现有数据处理算法重新扫描完整原文件：用 `SHA256("43\\0" + text.strip())` 的前 64 位分桶，验证比例 0.01；输出保留原始 text，并以原训练代码的 JSON 序列化方式写入。重建完整验证集的 SHA256 **与服务器 run_manifest 的 val_path SHA256 完全相同**，验证集共 12,560 条。

从完整验证集用 reservoir sampling 抽取 256 条，独立抽样种子 20261013。完整集和子集均重新用同一 tokenizer 编码，正文截断到 338 token，外加 BOS/EOS，右侧 padding 到 340；labels 在损失函数中移位一次，padding 标签为 -100。

''' + table(['对象','SHA256'], [
    ['原始语料',r['split']['raw_sha256']],
    ['tokenizer.json',r['manifest']['data_sha256']['tokenizer_path']],
    ['原训练集（服务器 manifest 记录，本次未重建）',r['manifest']['data_sha256']['train_path']],
    ['完整验证集（本次已重建并核对）',r['split']['full_val_sha256']],
    ['256 条验证子集',r['split']['subset_sha256']],
    ['checkpoint.pt（实验前后相同）',r['checkpoint_sha256']],
]) + '''

限制：划分保证归一化后完全相同的文本归属于同一集合，未检验近似重复或截断后 token 重叠。固定生成题没有检查是否在训练语料中出现，不能作为严格证明泛化能力的隐藏测试集。数据许可信息沿用 `dataset/SOURCE.json`，没有重新发布语料。

## 4. 数值质量测试

### 4.1 服务器历史记录

下面是历史训练日志中的指标，非今天重新训练所得。验证 loss 为移位后有效 token 的交叉熵均值；PPL = exp(loss)，自然对数。训练 loss 是一轮内各次更新前损失的 token 加权平均，不能当作最终固定模型在训练集上的 loss。

''' + table(['阶段','optimizer step','训练 loss','完整验证 loss','验证 PPL'], [
    ['训练前',0,'—',f"{r['historical_metrics'][0]['validation']['loss']:.6f}",f"{math.exp(r['historical_metrics'][0]['validation']['loss']):.4f}"],
    *[[f"epoch {x['completed_epochs']}",x['global_step'],f"{x['train']['loss']:.6f}",f"{x['validation']['loss']:.6f}",f"{math.exp(x['validation']['loss']):.4f}"] for x in r['historical_metrics'][1:]]
]) + '''

验证目标 token 数为 2,625,554。两轮验证 loss 均有下降；这支持“当前验证语料上的预测能力改善”，不能单独证明事实正确或没有过拟合。第 2 轮训练及验证耗时 5771.299 秒，历史峰值 allocated 为 7952.58 MiB；它们属于服务器训练，不能与下方本地推理的速度或显存直接混比。

### 4.2 今天重新测量的验证结果

本地 FP32、eval + inference/no_grad、batch_size=8、无梯度更新。完整集与子集独立报告，使用有效 token 加权交叉熵。

''' + table(['范围','样本','有效目标 token','batch 数','loss','PPL','耗时秒','峰值 allocated MiB'], [
    ['完整验证集',full['records'],f"{full['valid_tokens']:,}",full['steps'],f"{full['loss']:.6f}",f"{full['ppl']:.4f}",f"{full['elapsed_seconds']:.3f}",f"{full['peak_allocated_mib']:.2f}"],
    ['固定验证子集',256,f"{r['local_validation']['valid_tokens']:,}",r['local_validation']['steps'],f"{r['local_validation']['loss']:.6f}",f"{r['local_validation']['ppl']:.4f}",f"{r['local_validation']['elapsed_seconds']:.3f}",f"{r['local_validation']['peak_allocated_mib']:.2f}"]
]) + f'''

完整集 FP32 loss 与服务器 BF16 历史值的差为 {full['loss']-r['historical_metrics'][-1]['validation']['loss']:+.6f}。两者的 GPU、PyTorch、精度和 batch_size 不同，不要求逐位相等。子集是另一种样本集合，其较低 loss 不代表模型再次改善。

PPL 仅在分词器、语料、截断和损失口径可比时适合对照；不转换为通用能力评分或问答准确率。

## 5. 本地推理性能测试

### 5.1 环境与测量口径

''' + table(['项目','值'], [
    ['系统',r['environment']['platform']],
    ['Python',r['environment']['python'].split('|')[0].strip()],
    ['PyTorch / CUDA 构建',r['environment']['torch']+' / '+r['environment']['cuda_build']],
    ['tokenizers',r['environment']['tokenizers']],
    ['GPU / 总显存',r['environment']['gpu']+f" / {r['environment']['gpu_memory_bytes']/1024**3:.2f} GiB"],
    ['GPU 驱动','546.30（nvidia-smi 实测）'],
    ['计算方式','FP32，batch=1，全前缀重算；没有启用 KV cache'],
    ['CPU threads','2'],
    ['推理状态','eval + inference_mode；权重不更新'],
]) + '''

计时前后执行 CUDA synchronize；模型加载、tokenizer 初始化、输入编码、输出解码与控制台打印不计入生成循环。首 token 延迟包含首次前向、过滤、选 token 和同步，但不包含加载与编码。生成循环吞吐 = 实际输出 token 数 / 同步后的总循环秒数；分母含拼接、检查和结束 token 检测开销，**不是启用 KV cache 后的 decode 吞吐**。

性能测试前已用一条完整贪心生成预热，并核对带计时的实现与现有交互入口在该输入上的输出完全一致。每条生成仅测一次；延迟和吞吐属于这次机器状态下的观测值，没有稳定性置信区间。不同输出长度和 EOS 时机会影响吞吐。

### 5.2 全序列前向延迟

每种长度使用固定随机合法 token、batch=1，预热 5 次，再测 20 次。不含采样和自回归循环，不代表自然文本质量。

''' + table(['输入 token 数','中位数 ms','均值 ms','最小 ms','最大 ms','峰值 allocated MiB'], [
    [x['input_tokens'],f"{x['median_ms']:.3f}",f"{x['mean_ms']:.3f}",f"{x['min_ms']:.3f}",f"{x['max_ms']:.3f}",f"{x['peak_allocated_mib']:.2f}"] for x in r['forward_benchmark']
]) + '''

### 5.3 固定文本生成性能汇总

''' + table(['模式','用例数','输出 token 合计','整体 token/s','首 token 延迟中位数 ms','峰值 allocated MiB','EOS 次数','到 96 token 上限次数','重复 4-gram 比例'], [mode_summary('greedy'),mode_summary('sample')]) + '''

整体 token/s 使用 token 总数除以各条生成耗时总和，不是逐条 token/s 的算术平均。重复 4-gram 比例定义为 `1 - 不同连续4-token片段数 / 全部连续4-token片段数`，汇总按片段数量加权；它是重复诊断，不是语义正确率。所有用例均未触及 340-token 上下文上限；达到 96 新 token 上限表示人为截断，不证明回答自然完成。

allocated 指 PyTorch 当前活动张量的峰值，含模型与 RoPE buffer；不等于显卡全部进程显存。生成过程最大 reserved 为 ''' + f"{max(x['peak_reserved_mib'] for x in r['generations']):.2f}" + ''' MiB，可能复用了先前验证的缓存，不能当作生成所必需的张量显存。

环境警告：`Torch was not compiled with flash attention`。代码调用 SDPA 后由当前环境选择可用后端；本次没有测定具体替代后端。该警告与性能有关，不能据此解释事实编造或证明模型权重异常。

## 6. 固定生成用例及原始输出

- 输入格式：BOS + 原始文本；无聊天模板、无输入末尾 EOS，每条独立，不能使用上一条对话信息。
- greedy：temperature=0，argmax，top_k 不参与选择。
- sample：temperature=0.8，top_k=50；每条重新设置 seed=42，所以不依赖测试题顺序。
- 最大新增 token=96；PAD 与 BOS 不允许被生成，遇 EOS 停止；没有额外重复惩罚。
- 输出是输入后的续写，保留模型原始内容，没有人工纠错或美化。
- 以下 10 题不是公开基准，不能汇总成 HumanEval / C-Eval / 通用准确率。

''']

comments = {
 'P01':'概念续写整体通顺；采样末尾出现转向“以上文本内容”的未完成片段。',
 'P02':'贪心重复创建者与语言特性；采样仍有冗余，但语言连贯性较好。未对该题所有历史事实逐条审计。',
 'P03':'两种方式均编造或混淆学校信息；贪心还重复“斯坦福大学”。电子科技大学位于四川成都、始建于1956年，参见[学校官方简介](https://www.uestc.edu.cn/xxgk/xxjj.htm)。',
 'P04':'贪心给出100摄氏度；采样新增错误温标换算，并将压力与温度混淆，不能判为完整回答正确。',
 'P05':'贪心重复“把它放进”而非拿出所需操作；采样偏离取物任务，不能根据上下文给出直接操作。',
 'P06':'只提取首个 add 函数进行实际编译测试；贪心7个断言通过，采样因未声明c编译失败，见第7节。长解释及后续截断内容未作为另一段完整程序执行。',
 'P07':'贪心陷入相同动作的循环；采样出现“劫匪匪伯”等异常表达，故事因果关系薄弱。',
 'P08':'两种方式开头均未复述已给出的姓名。采样后文出现曹浩田，不能弥补其将“我的名字”填成另一人的错误。',
 'P09':'提到散射概念，但贪心后句自相矛盾；采样的日出日落与散射颜色解释不可靠。',
 'P10':'贪心开头概念基本相关，随后重复相同句子；采样出现“machine learning是machine learning的子集”的循环定义。',
}
for pid, category, prompt in [(x['id'],x['category'],x['prompt']) for x in r['generations'] if x['mode']=='greedy']:
    parts.append(f'### {pid} — {category}\n\n输入：\n\n~~~~text\n{prompt}\n~~~~\n\n')
    for x in [x for x in r['generations'] if x['id']==pid]:
        parts.append(f"**{x['mode']} 原始续写**（输入 {x['prompt_tokens']} token；输出 {x['generated_tokens']} token；停止原因 `{x['stop_reason']}`）：\n\n~~~~text\n{x['output']}\n~~~~\n\n")
        parts.append(f"耗时 {x['elapsed_seconds']:.3f} s；吞吐 {x['output_tokens_per_second']:.2f} token/s；首 token {x['first_token_seconds']*1000:.2f} ms；重复 4-gram {x['repeated_token_4gram_fraction']:.2%}。\n\n")
    parts.append('人工观察：'+comments[pid]+'\n\n')

parts.append('''## 7. 生成代码的功能检查

从 P06 每个输出中提取首个 `int add(int a, int b) {...}`，不更改函数体；分别用本地 `E:/Mingw/bin/g++.exe -std=c++17 -O2` 编译，并添加固定的 assert 测试入口。编译失败不执行。断言覆盖正数、零、负数、正负混合；没有测试整数溢出。

''' + table(['模式','编译返回码','执行','通过断言','结论'], [
    ['greedy',c['greedy']['compile_returncode'],'已执行',f"{c['greedy']['passed_cases']}/7",'本题提取函数通过'],
    ['sample',c['sample']['compile_returncode'],'未执行','—','未声明变量 c，编译失败'],
]) + '\n\n' + table(['a','b','期望值'],c['greedy']['cases']) + '''

采样函数与编译错误关键内容：

~~~~cpp
int add(int a, int b) {return a + b + c;}
~~~~

~~~~text
error: 'c' was not declared in this scope
~~~~

只有一个代码题、两个解码设置，不能据此估计通用编程通过率或 pass@k。

## 8. 下一步任务与验收标准

按先确认实现、再微调、再评测和优化的顺序推进。下面均为待完成任务；本次日志没有声称已运行它们。

| 优先级 | 待完成内容 | 验收成果与需要理解的内容 |
| --- | --- | --- |
| P0 | 对照固定 MiniMind 提交，核验模型数值实现 | 小模型加载相同映射权重、相同输入；逐层比较 hidden/logits、移位loss、关键梯度；覆盖padding和无padding，记录精度与误差容限；区分有意改动与bug。仅通过现有形状测试不替代这个检查。 |
| P0 | 冻结本次 Base 基线与评测输入 | 保留当前checkpoint及SHA、原始输出和实验脚本；后续SFT使用相同题、种子与长度；记录输入格式差异。 |
| P1 | 实现SFT数据处理 | 核对上游SFT数据来源与许可；统一聊天模板，明确assistant回答与结束标记监督，user/system/padding标签=-100；截断后无有效回答的样本明确处理；逐token检查标签边界，防止移位两次。 |
| P1 | 从当前权重开展全参数SFT | 从Base加载模型，使用新的SFT优化器和调度状态；先小批数据拟合，确认只监督回答，再完整训练；保持独立验证集，保存配置、曲线与checkpoint。SFT有望改善指令与回答格式，不保证补齐事实知识。 |
| P1 | 比较Base与SFT并扩展评测 | 相同问题保留完整输出；比较指令执行、上下文保持、重复、结束行为及实际代码功能；预训练PPL与SFT回答loss分开报告；数据集和监督口径不同的loss不直接横比。 |
| P2 | 启用并验证KV cache | 比较全前缀与cache推理的logits/生成一致性，再固定长度和精度测TTFT、decode吞吐、显存，避免只比较不同长度的文本。 |
| P2 | 根据失败样本安排专项改进 | 检查语料和截断影响；一次改变一个因素，评估额外训练或专项微调的效果；具体任务与数据另行确定。Agent接口集成放在模型基线明确之后。 |

固定参考提交：`f659b55761b754d306bd140573493a6543cafd7f`。[模型源码](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/model/model_minimind.py)、[SFT数据源码](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/dataset/lm_dataset.py)。参考上游仅用于核对，正式运行继续使用TrainPilot自己的实现。

## 9. 复现与原始记录

从项目根目录执行本地实验；它读取既有权重并生成验证文件、JSON与代码检查产物，不执行训练：

~~~~powershell
cd E:\\agent项目\\trainpilot
& .\\.venv\\Scripts\\python.exe -u test/baseline_20261013/evaluate_baseline.py
& .\\.venv\\Scripts\\python.exe test/baseline_20261013/check_generated_cpp.py
& .\\.venv\\Scripts\\python.exe test/baseline_20261013/write_report.py
~~~~

第一次完整验证使用 `complete_full_validation.py` 对已完成的生成实验追加验证；当前 `evaluate_baseline.py` 已包含同一完整验证流程，复现无需重复运行补充脚本。耗时与吞吐受GPU负载、驱动和温度影响，不承诺重新运行数值完全相同。

已提交的原始记录：[results.json](base_model_20261013/results.json) 与 [code_results.json](base_model_20261013/code_results.json)。文件内保留的是当时测量的配置、路径和结果。

可执行的复现脚本放在 `test/baseline_20261013/`；从原本本地实验脚本整理而来，路径定位及编译器查找方式已调整，未重新测量或替换已记录结果。原始本地产物位于 `runs/base-eval-20261013/`（runs按仓库规则忽略）：

- `results.json`：配置、环境、原始生成token ID/文本、loss/PPL、延迟、显存、SHA及停止原因。
- `code_results.json`：编译命令、编译/执行返回码、stderr和断言用例。
- `validation_full.jsonl` / `validation_subset.jsonl`：用于今天验证的重建数据。
- `generated_add_greedy.cpp` / `generated_add_sample.cpp`：提取函数及断言入口。
- 原运行脚本保留在 runs；仓库复现入口位于 `test/baseline_20261013/`。

该Markdown可以单独阅读；完整复现实验还需本地权重、语料和上述原始产物。GitHub仅保存日志、原始结果JSON和复现代码，不保存本次权重、验证文本或二进制程序。用户指定的本地文件名 `test_base_model_10.13,md` 保留；GitHub阅读版本为 `test_base_model_2026-10-13.md`。

运行代码的文件指纹（日志生成时读取）：

''' + table(['文件','SHA256'], [[str(p.relative_to(ROOT)).replace('\\','/'),digest(p)] for p in [
    ROOT/'test/base_model_test.py',ROOT/'trainpilot/config.py',ROOT/'trainpilot/model.py',
    ROOT/'trainpilot/layers.py',ROOT/'trainpilot/data.py',ROOT/'trainpilot/pretrain.py',
    ROOT/'trainpilot/trainer.py',ROOT/'test/baseline_20261013/evaluate_baseline.py',ROOT/'test/baseline_20261013/check_generated_cpp.py'
]]) + '\n')

TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(''.join(parts), encoding='utf-8')
TARGET.with_name('test_base_model_2026-10-13.md').write_text(''.join(parts), encoding='utf-8')
assert '\ufffd' not in TARGET.read_text(encoding='utf-8'), 'replacement characters found'
print('report saved; bytes=',TARGET.stat().st_size)
