# TrainPilot 基础模型测试日志 — 2026-10-13

## 1. 实验信息与结论

- 实验编号：BASE-20261013-01。
- 模型阶段：完成两轮预训练的 Dense 基础语言模型；本次没有执行 SFT 或更新参数。
- 生成实验记录时间：2026-10-13T22:38:40.830295+08:00（北京时间）。完整验证复测在同一晚随后完成。
- 权重基线：`runs/pre_train1-20261012/checkpoint.pt`。
- 测试对象：63,912,192 个独立参数，约 64M。
- 当前代码 HEAD：`c586fe05a5695d7ed9ab9b6610e6c3c2b06a0ca8`。工作树含本地修改；HEAD 本身不能唯一标识本次运行的所有文件，具体代码 SHA256 见第 9 节。训练日志未记录训练时 Git SHA，不能据此证明训练与当前源码逐字一致。
- 记录来源：服务器保存的 `metrics.jsonl` / `run_manifest.json`，以及今天实际运行的本地实验。历史人工对话仅用于选择测试题，下面的输出全部为本次重新生成。

**结论：模型已学习到部分语言和概念表达，但当前的事实可靠性、上下文保持和生成稳定性不足。** 完整验证复测 loss=1.924007、PPL=6.8483；这些指标不代表问答准确率。概念续写较流畅，但固定用例暴露了大学信息编造、姓名保持失败及重复循环。简单 C++ 加法函数在贪心生成下编译并通过 7 个断言；采样生成引入未声明变量，编译失败。

本次没有运行 MiniMind 对照模型，没有测得与其质量差距或数值一致性；不能宣称已达到或落后于某个官方权重的定量水平。

## 2. 模型与训练基线

| 项目 | 配置 |
| --- | --- |
| 模型结构 | Dense / Pre-Norm / RMSNorm / RoPE / GQA / Q-K Norm / SwiGLU / 共享输入输出嵌入 |
| 词表 / 层数 / 隐藏维度 | 6400 / 8 / 768 |
| Q 头 / KV 头 / 每头维度 | 8 / 4 / 96 |
| FFN 中间维度 | 2432 |
| 训练截断长度 / RoPE buffer 上限 | 340 / 32768；buffer 上限不代表模型已学会长上下文 |
| 训练精度 / epoch | BF16 autocast / 2 |
| 优化器 | AdamW；lr=0.0005，weight_decay=0.01 |
| 梯度累积 / 裁剪 | batch_size=32，accumulation_steps=8，有效 batch 约 256，clip=1.0 |
| 调度 / 训练种子 | 余弦衰减至初始 LR 的 0.1 / seed=42 |
| 完成的优化器更新数 | 9826 |
| 每轮有效训练目标 token | 262,273,758；两轮累计 524,547,516，包含重复遍历 |
| 训练服务器软件 | Python 3.12.3 / PyTorch 2.8.0+cu128 / CUDA 构建 12.8 |
| 训练 GPU | RTX 4090，来自此前服务器环境记录；run_manifest 本身没有保存 GPU 型号 |
| checkpoint 文件大小 | 767,069,419 bytes / 731.53 MiB；包含优化器等状态，不是纯权重大小 |

## 3. 数据来源、划分与校验

原始语料为 `jingyaogong/minimind_dataset` 的 `pretrain_t2t_mini.jsonl`，固定 revision `312afb4f76391145c6902f765bb51691c09a12f5`。本地原文件的 SHA256 与 `dataset/SOURCE.json` 一致。

本次按现有数据处理算法重新扫描完整原文件：用 `SHA256("43\0" + text.strip())` 的前 64 位分桶，验证比例 0.01；输出保留原始 text，并以原训练代码的 JSON 序列化方式写入。重建完整验证集的 SHA256 **与服务器 run_manifest 的 val_path SHA256 完全相同**，验证集共 12,560 条。

从完整验证集用 reservoir sampling 抽取 256 条，独立抽样种子 20261013。完整集和子集均重新用同一 tokenizer 编码，正文截断到 338 token，外加 BOS/EOS，右侧 padding 到 340；labels 在损失函数中移位一次，padding 标签为 -100。

| 对象 | SHA256 |
| --- | --- |
| 原始语料 | 6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c |
| tokenizer.json | 71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668 |
| 原训练集（服务器 manifest 记录，本次未重建） | 691c3979fbac91cca525939787dca05c5f6769187b6adbab7e433305607c514b |
| 完整验证集（本次已重建并核对） | 16cc01484d544f8914594ea8e81235b000b74d7a39739296e95734a3e62ceb2a |
| 256 条验证子集 | c214716b0fe248322398ad4187fc923fbb591dc258a3e64794ca98fbba64adb7 |
| checkpoint.pt（实验前后相同） | 13c61a9845aeb4d7f24cb2ac6109cf161270e74693b70f214b61561eb75f52ec |

限制：划分保证归一化后完全相同的文本归属于同一集合，未检验近似重复或截断后 token 重叠。固定生成题没有检查是否在训练语料中出现，不能作为严格证明泛化能力的隐藏测试集。数据许可信息沿用 `dataset/SOURCE.json`，没有重新发布语料。

## 4. 数值质量测试

### 4.1 服务器历史记录

下面是历史训练日志中的指标，非今天重新训练所得。验证 loss 为移位后有效 token 的交叉熵均值；PPL = exp(loss)，自然对数。训练 loss 是一轮内各次更新前损失的 token 加权平均，不能当作最终固定模型在训练集上的 loss。

| 阶段 | optimizer step | 训练 loss | 完整验证 loss | 验证 PPL |
| --- | --- | --- | --- | --- |
| 训练前 | 0 | — | 8.934648 | 7590.4633 |
| epoch 1 | 4913 | 2.483786 | 2.067029 | 7.9013 |
| epoch 2 | 9826 | 1.927019 | 1.924127 | 6.8492 |

验证目标 token 数为 2,625,554。两轮验证 loss 均有下降；这支持“当前验证语料上的预测能力改善”，不能单独证明事实正确或没有过拟合。第 2 轮训练及验证耗时 5771.299 秒，历史峰值 allocated 为 7952.58 MiB；它们属于服务器训练，不能与下方本地推理的速度或显存直接混比。

### 4.2 今天重新测量的验证结果

本地 FP32、eval + inference/no_grad、batch_size=8、无梯度更新。完整集与子集独立报告，使用有效 token 加权交叉熵。

| 范围 | 样本 | 有效目标 token | batch 数 | loss | PPL | 耗时秒 | 峰值 allocated MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 完整验证集 | 12560 | 2,625,554 | 1570 | 1.924007 | 6.8483 | 144.798 | 477.05 |
| 固定验证子集 | 256 | 55,252 | 32 | 1.911207 | 6.7612 | 2.929 | 477.10 |

完整集 FP32 loss 与服务器 BF16 历史值的差为 -0.000120。两者的 GPU、PyTorch、精度和 batch_size 不同，不要求逐位相等。子集是另一种样本集合，其较低 loss 不代表模型再次改善。

PPL 仅在分词器、语料、截断和损失口径可比时适合对照；不转换为通用能力评分或问答准确率。

## 5. 本地推理性能测试

### 5.1 环境与测量口径

| 项目 | 值 |
| --- | --- |
| 系统 | Windows-10-10.0.22631-SP0 |
| Python | 3.10.13 |
| PyTorch / CUDA 构建 | 2.2.1+cu118 / 11.8 |
| tokenizers | 0.22.1 |
| GPU / 总显存 | NVIDIA GeForce RTX 3060 Laptop GPU / 6.00 GiB |
| GPU 驱动 | 546.30（nvidia-smi 实测） |
| 计算方式 | FP32，batch=1，全前缀重算；没有启用 KV cache |
| CPU threads | 2 |
| 推理状态 | eval + inference_mode；权重不更新 |

计时前后执行 CUDA synchronize；模型加载、tokenizer 初始化、输入编码、输出解码与控制台打印不计入生成循环。首 token 延迟包含首次前向、过滤、选 token 和同步，但不包含加载与编码。生成循环吞吐 = 实际输出 token 数 / 同步后的总循环秒数；分母含拼接、检查和结束 token 检测开销，**不是启用 KV cache 后的 decode 吞吐**。

性能测试前已用一条完整贪心生成预热，并核对带计时的实现与现有交互入口在该输入上的输出完全一致。每条生成仅测一次；延迟和吞吐属于这次机器状态下的观测值，没有稳定性置信区间。不同输出长度和 EOS 时机会影响吞吐。

### 5.2 全序列前向延迟

每种长度使用固定随机合法 token、batch=1，预热 5 次，再测 20 次。不含采样和自回归循环，不代表自然文本质量。

| 输入 token 数 | 中位数 ms | 均值 ms | 最小 ms | 最大 ms | 峰值 allocated MiB |
| --- | --- | --- | --- | --- | --- |
| 16 | 9.065 | 10.440 | 7.982 | 16.814 | 278.77 |
| 64 | 9.074 | 9.304 | 8.431 | 13.139 | 280.53 |
| 256 | 9.936 | 10.613 | 8.855 | 15.877 | 288.56 |

### 5.3 固定文本生成性能汇总

| 模式 | 用例数 | 输出 token 合计 | 整体 token/s | 首 token 延迟中位数 ms | 峰值 allocated MiB | EOS 次数 | 到 96 token 上限次数 | 重复 4-gram 比例 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greedy | 10 | 720 | 97.61 | 9.95 | 282.79 | 5 | 5 | 40.58% |
| sample | 10 | 830 | 93.52 | 10.39 | 282.82 | 4 | 6 | 4.37% |

整体 token/s 使用 token 总数除以各条生成耗时总和，不是逐条 token/s 的算术平均。重复 4-gram 比例定义为 `1 - 不同连续4-token片段数 / 全部连续4-token片段数`，汇总按片段数量加权；它是重复诊断，不是语义正确率。所有用例均未触及 340-token 上下文上限；达到 96 新 token 上限表示人为截断，不证明回答自然完成。

allocated 指 PyTorch 当前活动张量的峰值，含模型与 RoPE buffer；不等于显卡全部进程显存。生成过程最大 reserved 为 670.00 MiB，可能复用了先前验证的缓存，不能当作生成所必需的张量显存。

环境警告：`Torch was not compiled with flash attention`。代码调用 SDPA 后由当前环境选择可用后端；本次没有测定具体替代后端。该警告与性能有关，不能据此解释事实编造或证明模型权重异常。

## 6. 固定生成用例及原始输出

- 输入格式：BOS + 原始文本；无聊天模板、无输入末尾 EOS，每条独立，不能使用上一条对话信息。
- greedy：temperature=0，argmax，top_k 不参与选择。
- sample：temperature=0.8，top_k=50；每条重新设置 seed=42，所以不依赖测试题顺序。
- 最大新增 token=96；PAD 与 BOS 不允许被生成，遇 EOS 停止；没有额外重复惩罚。
- 输出是输入后的续写，保留模型原始内容，没有人工纠错或美化。
- 以下 10 题不是公开基准，不能汇总成 HumanEval / C-Eval / 通用准确率。

### P01 — 概念续写

输入：

~~~~text
人工智能是一种
~~~~

**greedy 原始续写**（输入 3 token；输出 66 token；停止原因 `eos`）：

~~~~text
模拟人类智能的技术，它可以让计算机系统自主学习、推理和决策，从而实现自主决策和执行任务。人工智能技术已经广泛应用于各个领域，包括医疗、金融、交通、制造业等。它可以帮助我们更好地理解和解决复杂的问题，提高生产力和效率，改善生活质量。
~~~~

耗时 0.744 s；吞吐 88.65 token/s；首 token 12.73 ms；重复 4-gram 0.00%。

**sample 原始续写**（输入 3 token；输出 96 token；停止原因 `max_new_tokens`）：

~~~~text
模拟、延伸和扩展人类认知和行为的技术和方法。它的应用范围已经从最初的计算机程序和算法转移到了更广泛的领域，如自然语言处理、计算机视觉和机器学习等。人工智能已经深入到我们生活的各个领域，如医药、金融、制造业等。它不仅提高了生产效率和工作效率，还带来了许多新的商业机会和创新，未来发展的前景将更加广阔。基于以上这段文本内容
~~~~

耗时 1.146 s；吞吐 83.76 token/s；首 token 144.06 ms；重复 4-gram 0.00%。

人工观察：概念续写整体通顺；采样末尾出现转向“以上文本内容”的未完成片段。

### P02 — 解释指令

输入：

~~~~text
介绍一下python。
~~~~

**greedy 原始续写**（输入 5 token；输出 96 token；停止原因 `max_new_tokens`）：

~~~~text
Python是一种高级编程语言，由Guido van Rossum于1989年创建。它是一种解释型语言，由Guido van Rossum于1989年创建。Python具有简单易学、可读性强、可扩展性强等特点，被广泛应用于Web开发、科学计算、人工智能、机器学习等领域。Python语言具有简单易学、可读性强、可扩展性强等特点，
~~~~

耗时 0.970 s；吞吐 98.99 token/s；首 token 12.23 ms；重复 4-gram 31.18%。

**sample 原始续写**（输入 5 token；输出 77 token；停止原因 `eos`）：

~~~~text
Python是一种高级语言，属于解释型的程序语言，有大量的库和框架可供选择，可以用于各种应用程序开发。Python具有简单易学的语法和强大的功能，适合用于数据分析、机器学习、Web开发、人工智能、科学计算等各种领域。Python还拥有丰富的库和模块，可以用于科学计算、数据分析、机器学习、大数据处理、机器学习等多个领域。
~~~~

耗时 0.793 s；吞吐 97.09 token/s；首 token 10.21 ms；重复 4-gram 1.35%。

人工观察：贪心重复创建者与语言特性；采样仍有冗余，但语言连贯性较好。未对该题所有历史事实逐条审计。

### P03 — 具体事实

输入：

~~~~text
介绍一下电子科技大学。
~~~~

**greedy 原始续写**（输入 7 token；输出 96 token；停止原因 `max_new_tokens`）：

~~~~text
电子科技大学是一所位于美国加州的综合性大学，成立于1911年，总部位于美国加州的西雅图市。该大学由哈佛大学、麻省理工学院、斯坦福大学、斯坦福大学、斯坦福大学、斯坦福大学、斯坦福大学、斯坦福大学、斯坦福大学、斯坦福大学、
~~~~

耗时 0.974 s；吞吐 98.55 token/s；首 token 9.37 ms；重复 4-gram 45.16%。

**sample 原始续写**（输入 7 token；输出 96 token；停止原因 `max_new_tokens`）：

~~~~text
电子科技大学是一所位于美国加州哈佛大学的一所综合性学术机构，其最著名的机构是哈佛大学。哈佛大学提供多种科学和工程科学课程，涵盖了计算机科学、数学、生物学、化学和物理学等多个领域。哈佛大学每年平均收成100,000美元，主要用于科学研究和工程相关研究。它还提供许多高等教育和研究项目，包括科学研究和工程
~~~~

耗时 1.079 s；吞吐 89.01 token/s；首 token 9.30 ms；重复 4-gram 2.15%。

人工观察：两种方式均编造或混淆学校信息；贪心还重复“斯坦福大学”。电子科技大学位于四川成都、始建于1956年，参见[学校官方简介](https://www.uestc.edu.cn/xxgk/xxjj.htm)。

### P04 — 基础事实

输入：

~~~~text
在标准大气压下，水的沸点是
~~~~

**greedy 原始续写**（输入 13 token；输出 13 token；停止原因 `eos`）：

~~~~text
多少？水的沸点是100摄氏度。
~~~~

耗时 0.128 s；吞吐 101.55 token/s；首 token 11.45 ms；重复 4-gram 0.00%。

**sample 原始续写**（输入 13 token；输出 61 token；停止原因 `eos`）：

~~~~text
多少？水的沸点大约是100摄氏度（83华氏度），但在标准大气压下（0摄氏度）。这表示在水中，水的压力约为91.8摄氏度（83华氏度）。
~~~~

耗时 0.632 s；吞吐 96.45 token/s；首 token 9.92 ms；重复 4-gram 17.24%。

人工观察：贪心给出100摄氏度；采样新增错误温标换算，并将压力与温度混淆，不能判为完整回答正确。

### P05 — 上下文保持

输入：

~~~~text
小林把红色笔记本放进蓝色背包。他想拿出笔记本，应该先
~~~~

**greedy 原始续写**（输入 27 token；输出 96 token；停止原因 `max_new_tokens`）：

~~~~text
把它放进蓝色背包里。他应该先把它放进蓝色背包里。他应该先把它放进蓝色背包里。他应该先把它放进蓝色背包里。他应该先把它放进蓝色背包里。他应该先把它放进蓝色背包里。他应该先把它放进蓝色背包里。他应该先把它放进蓝
~~~~

耗时 1.036 s；吞吐 92.66 token/s；首 token 8.89 ms；重复 4-gram 86.02%。

**sample 原始续写**（输入 27 token；输出 96 token；停止原因 `max_new_tokens`）：

~~~~text
放在哪里？根据题目描述，小林想要把红色笔记本放进蓝色背包，因此我们需要找到一个正确的答案，即“蓝色背包”。蓝色背包通常用于户外活动，如野餐，露营，徒步旅行等，因为它们适合在阳光充足的时候使用，因为它们很容易被太阳的热量和光线吸收。因此，无法找到蓝色背包。但是
~~~~

耗时 0.988 s；吞吐 97.14 token/s；首 token 9.68 ms；重复 4-gram 3.23%。

人工观察：贪心重复“把它放进”而非拿出所需操作；采样偏离取物任务，不能根据上下文给出直接操作。

### P06 — 简单代码

输入：

~~~~text
请用C++写一个函数 int add(int a, int b)，返回两个整数的和。
~~~~

**greedy 原始续写**（输入 23 token；输出 96 token；停止原因 `max_new_tokens`）：

~~~~text
好的，以下是用C++编写的函数，它将两个整数相加并返回它们的和：```cpp
int add(int a, int b) {
    return a + b;
}
```这个函数的名字是 `add`，它有两个参数 `a` 和 `b`，它们是两个整数。函数体内只有一行代码，它将这两个整数相加并返回它们的和。你可以像这样调用这个函数：```cpp
~~~~

耗时 0.916 s；吞吐 104.82 token/s；首 token 10.03 ms；重复 4-gram 5.38%。

**sample 原始续写**（输入 23 token；输出 96 token；停止原因 `max_new_tokens`）：

~~~~text
好的，我将使用C++编写一个简单的程序来接收用户输入的数字，并将它们相加，然后返回总和。以下是代码示例：```c++int add(int a, int b) {return a + b + c;}```代码中，我们首先定义了一个数字参数a和b，并将它们相加。最后，我们将结果存储在变量中。在函数内部，我们定义了一个数字变量b并将它
~~~~

耗时 1.010 s；吞吐 95.02 token/s；首 token 12.09 ms；重复 4-gram 3.23%。

人工观察：只提取首个 add 函数进行实际编译测试；贪心7个断言通过，采样因未声明c编译失败，见第7节。长解释及后续截断内容未作为另一段完整程序执行。

### P07 — 故事续写

输入：

~~~~text
清晨，小林走到公交站，突然发现钱包不见了，于是
~~~~

**greedy 原始续写**（输入 23 token；输出 96 token；停止原因 `max_new_tokens`）：

~~~~text
他拿起了钱包，开始寻找钱包。他走到公交站的门口，发现钱包不见了。他走到公交站的门口，发现钱包不见了。他走到公交站的门口，发现钱包不见了。他走到公交站的门口，发现钱包不见了。他走到公交站的门口，发现钱包不见
~~~~

耗时 0.941 s；吞吐 102.02 token/s；首 token 9.88 ms；重复 4-gram 68.82%。

**sample 原始续写**（输入 23 token；输出 96 token；停止原因 `max_new_tokens`）：

~~~~text
他就跑过去。在此时，他看到一道黑色的劫匪匪伯走了，于是他决定在离别区内尽情享受这道硬币。“你怎么了，这道硬币掉在地上了？”小林尖叫起来，惊讶地看着他，惊奇地注意到他的手心里一丝不安，就跑进了
~~~~

耗时 0.989 s；吞吐 97.10 token/s；首 token 12.18 ms；重复 4-gram 1.08%。

人工观察：贪心陷入相同动作的循环；采样出现“劫匪匪伯”等异常表达，故事因果关系薄弱。

### P08 — 前缀信息保持

输入：

~~~~text
我叫曹浩田。我的名字是
~~~~

**greedy 原始续写**（输入 13 token；输出 43 token；停止原因 `eos`）：

~~~~text
李华。我是一名学生。我喜欢读书。我喜欢读书。我喜欢读书。我喜欢读书。我喜欢读书。我喜欢读书。我喜欢读书。
~~~~

耗时 0.457 s；吞吐 94.09 token/s；首 token 9.17 ms；重复 4-gram 70.00%。

**sample 原始续写**（输入 13 token；输出 41 token；停止原因 `eos`）：

~~~~text
张大公。我在公司工作的程序员。我的职业是软件工程师。我最喜欢的职业是软件工程师。曹浩田是我职业的职业。
~~~~

耗时 0.420 s；吞吐 97.64 token/s；首 token 10.56 ms；重复 4-gram 7.89%。

人工观察：两种方式开头均未复述已给出的姓名。采样后文出现曹浩田，不能弥补其将“我的名字”填成另一人的错误。

### P09 — 事实解释

输入：

~~~~text
天空看起来是蓝色的，因为
~~~~

**greedy 原始续写**（输入 10 token；输出 59 token；停止原因 `eos`）：

~~~~text
太阳光线照射到地球上，所以我们看到的天空是蓝色的。这是因为太阳光线在穿过大气层时被散射，使得蓝色光线被散射得更多，而其他颜色的光线则被散射得更多。
~~~~

耗时 0.628 s；吞吐 93.88 token/s；首 token 9.47 ms；重复 4-gram 3.57%。

**sample 原始续写**（输入 10 token；输出 96 token；停止原因 `max_new_tokens`）：

~~~~text
天空是蓝色的。是的，天空看起来是蓝色的。这是因为大气层对太阳光的散射作用，在离子中会散射蓝色光的颜色。在日出和日落时，我们看到的天空呈现出蓝色的色调，这是因为大气层中的气体分子会散射太阳光中红色和黄色光线。这些颜色的光波长
~~~~

耗时 0.954 s；吞吐 100.58 token/s；首 token 9.53 ms；重复 4-gram 4.30%。

人工观察：提到散射概念，但贪心后句自相矛盾；采样的日出日落与散射颜色解释不可靠。

### P10 — 英文续写

输入：

~~~~text
Machine learning is a
~~~~

**greedy 原始续写**（输入 7 token；输出 59 token；停止原因 `eos`）：

~~~~text
 subset of artificial intelligence (AI) that can help automatically learn from data and predict future events. It can help automatically learn from data and predict future events. It can help automatically learn from data and predict future events.
~~~~

耗时 0.582 s；吞吐 101.43 token/s；首 token 10.46 ms；重复 4-gram 53.57%。

**sample 原始续写**（输入 7 token；输出 75 token；停止原因 `eos`）：

~~~~text
 subset of machine learning. It is designed to learn from a specific model that can provide information and assistance to the task of tasks. Machine learning is a subset of machine learning. It is used to develop data from a specific model that simulates tasks like image recognition, image recognition, or performing tasks that are used to understand the task.
~~~~

耗时 0.863 s；吞吐 86.92 token/s；首 token 12.26 ms；重复 4-gram 11.11%。

人工观察：贪心开头概念基本相关，随后重复相同句子；采样出现“machine learning是machine learning的子集”的循环定义。

## 7. 生成代码的功能检查

从 P06 每个输出中提取首个 `int add(int a, int b) {...}`，不更改函数体；分别用本地 `E:/Mingw/bin/g++.exe -std=c++17 -O2` 编译，并添加固定的 assert 测试入口。编译失败不执行。断言覆盖正数、零、负数、正负混合；没有测试整数溢出。

| 模式 | 编译返回码 | 执行 | 通过断言 | 结论 |
| --- | --- | --- | --- | --- |
| greedy | 0 | 已执行 | 7/7 | 本题提取函数通过 |
| sample | 1 | 未执行 | — | 未声明变量 c，编译失败 |

| a | b | 期望值 |
| --- | --- | --- |
| 1 | 2 | 3 |
| 0 | 0 | 0 |
| -1 | 1 | 0 |
| -2 | -3 | -5 |
| 100 | -50 | 50 |
| 123 | 456 | 579 |
| -1000 | 1000 | 0 |

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
cd E:\agent项目\trainpilot
& .\.venv\Scripts\python.exe -u test/test_code/baseline_20261013/evaluate_baseline.py
& .\.venv\Scripts\python.exe test/test_code/baseline_20261013/check_generated_cpp.py
& .\.venv\Scripts\python.exe test/test_code/baseline_20261013/write_report.py
~~~~

第一次完整验证使用 `complete_full_validation.py` 对已完成的生成实验追加验证；当前 `evaluate_baseline.py` 已包含同一完整验证流程，复现无需重复运行补充脚本。耗时与吞吐受GPU负载、驱动和温度影响，不承诺重新运行数值完全相同。

已提交的原始记录：[results.json](base_model_20261013/results.json) 与 [code_results.json](base_model_20261013/code_results.json)。文件内保留的是当时测量的配置、路径和结果。

可执行的复现脚本放在 `test/test_code/baseline_20261013/`；从原本本地实验脚本整理而来，路径定位及编译器查找方式已调整，未重新测量或替换已记录结果。原始本地产物位于 `runs/base-eval-20261013/`（runs按仓库规则忽略）：

- `results.json`：配置、环境、原始生成token ID/文本、loss/PPL、延迟、显存、SHA及停止原因。
- `code_results.json`：编译命令、编译/执行返回码、stderr和断言用例。
- `validation_full.jsonl` / `validation_subset.jsonl`：用于今天验证的重建数据。
- `generated_add_greedy.cpp` / `generated_add_sample.cpp`：提取函数及断言入口。
- 原运行脚本保留在 runs；仓库复现入口位于 `test/test_code/baseline_20261013/`。

该Markdown可以单独阅读；完整复现实验还需本地权重、语料和上述原始产物。GitHub仅保存日志、原始结果JSON和复现代码，不保存本次权重、验证文本或二进制程序。原逗号文件名副本已在2026-10-14按用户要求移到仓库外备份；GitHub阅读版本为 `test_base_model_2026-10-13.md`。

以下文件指纹保留2026-10-13测量时的原路径和SHA256；2026-10-14仅更新了可执行脚本、文档链接和目录位置，没有重新测量或改写历史数值。

运行代码的文件指纹（日志生成时读取）：

| 文件 | SHA256 |
| --- | --- |
| test/base_model_test.py | b1af51395641f139a7204438e7568f3f07d45b1b687c3e5dbe4587dc920f63e2 |
| trainpilot/config.py | 20cc61ce63b4a235736068608ddee8b05bf88f029c98f0a574c24cb634f37b83 |
| trainpilot/model.py | 0f0c04e87fec65ae4231a7d8bcc701cc3e7e14acbe535907aec58a7b68bb895c |
| trainpilot/layers.py | 83b90efca943e64af7795f34c49a22fb8e8a24986056076403b0078405c2be11 |
| trainpilot/data.py | 3d8539fdb3ab555a6674e5eced38b7d181fd7e514bd80a365dbe1ffb09d08ac3 |
| trainpilot/pretrain.py | 55cd5d192fc6000dd27852625061e3448ebdfa9bc21c9f64026ebebeebd450ff |
| trainpilot/trainer.py | 4f2cd4388e30198b4b75a7f905dcb0f61651805e9ba9b1304c6fe6302b2eac5b |
| test/baseline_20261013/evaluate_baseline.py | c0314e58988d6e46566200b095d7130d9d2cbcc653ef41d15d2e1ccbbd0fba06 |
| test/baseline_20261013/check_generated_cpp.py | 6d5c136b03cc007f09381b1a428fbeb52ec5c2988a904edbffdfb53ef29e1230 |
