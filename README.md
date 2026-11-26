# TrainPilot

本项目是基于minimind的自学习开发项目，目的是从零开始构建一个轻量级语言模型，为用户提供规模训练中的结果分析。本项目在考虑实用性的同时，更多的是模型构建的学习项目，从零开始对模型的训练过程进行拆分重构，并在原有基础上进行改进。本项目中可能涉及到学习过程的记录，可酌情参考。

## 项目结构如下

```text
trainpilot/
├─ model/                   # 模型训练与评测
│  ├─ network/              # 自行实现的 Dense Transformer
│  ├─ config/               # 训练与模型配置
│  ├─ dataset/              # 数据处理与训练语料
│  ├─ trainer/              # 预训练、Full SFT、LoRA、GRPO
│  ├─ serving/              # 本机权重加载与工具兼容 API
│  └─ test/                 # 模型测试代码与历史测试日志
├─ agent/                   # Agent 搭建与本地演示
│  ├─ app/                  # LangGraph、业务服务、权限和 HTTP 接口
│  ├─ enterprise_rag/       # 企业语料、索引、检索与证据校验
│  ├─ local_runtime/        # 本地页面、服务编排与诊断
│  ├─ config/               # 固定 V7 的部署配置
│  ├─ sql/                  # 演示数据库结构及迁移
│  └─ tests/                # Agent 测试
├─ runs/                    # 原权重与实验路径保持不变（不提交）
├─ images/
├─ 启动项目.cmd
└─ 停止项目.cmd
```

重排后的命令（在仓库根目录执行）：

```bash
python -m agent
python -m model.trainer.train --config model/config/smoke.json
python -m model.trainer.train --config model/config/pretrain.json
python -m model.test.test_code.base_model_test --checkpoint runs/pre_train1-20261012/checkpoint.pt
python -m unittest discover -s model/test/test_code -p "test_*.py" -v
```

## 10.10

编写了RMSNorm与SwiGLU，这两个模块作为整个模型中最基本的操作模块是很重要的，两个模块的作用已经写到了layers.py中，这个文件主要描述的是层级的基本操作

## 10.11

编写了RoPE位置编码的代码，block，model的实现，并与lm_head组成成为完整的casualllm模板。上传了代码与结构图（保留）。

## 10.12

完成了epoch训练完整流程，下载了语料库并且在云服务器上进行部署与训练，预期今天将训练出最基础的语言模型，当前训练使用的是minimind提供的语料库，训练两个epoch后查看效果

## 10.13

检查当前基础模型的性能，进行直观测试和数值测试，测试结果已经保存到路径test_log\test_base_model_10.13,md，并且罗列接下来的工作路线

## 10.14

进行基础模型的数值分析收尾。现有的基础模型只具备文本续写的能力，在问答方面的表现会非常的奇怪，当前阶段进行的SFT是大模型有监督微调，目的是让大模型学会以回答的形式回答用户问题而不是进行续写，训练结束后对于不像询问的提问模型也会进行标准的回答。首先进行SFT的数据集预处理，主要需要包括数据重构，数据解码，数据选择，数据预处理(选择哪些数据计算损失，哪些数据不关注)

编写了SFT_dataset对象，读取sft数据之后，进行了比如随机去除思考标签，无sys时随机添加标签的操作，并且将读取的sft数据转变为正常的python可读取对象。训练的时候只考虑asst产生的回答，所以只对这部分的输出进行训练，增加合适的padding标签来避免他的损失进入训练过程。最后返回'input_ids','attention_mask','labels'，与训练器的输入保持一致。和续写模型相比由于数据集的变化考虑了更多情况。

添加了额外的数据检查，防止出现截断之后的数据泄露与无效数据，并且划分了数据集和验证集，将train函数新增了sft的入口，做好正式训练前的准备

## 10.15

使用基础Qwen模型作为基础模型，在Qwen1.5B-base模型上进行全量SFT训练，得到训练结果后进行测试，由于更换模型，需要替换一些对象返回组成与构成形式，前向生成使用的是官方的架构。这里换模型主要是为了测试官方的轻量小模型的性能是否相比于我们有明显的提升。

## 10.21

测评full sft qwen3模型，发现效果良好，与原本的自训小模型有本质的差别。一是数据集官方规范的更标准，二是qwen3-1.5b使用了28层block来进行训练和生成，能够理解并且处理更加复杂的任务，训练相关的数据已经进行了保存，训练时随着loss减少的速度变缓逐步的减小学习率，避免模型在优解附近震荡。接下来将进行工具调用与思考的lora训练，需要根据agent端产出的小demo运行完整的闭环。

## 10.30

基于Qwen3-1.7B完成企业业务LoRA SFT与V6修复训练，统一Agent提示词，通过状态对照、字段保留和固定证据RAG评测定位错误。发现旧业务修复不等于新表达泛化，进一步用多次采样区分“不能生成正确答案”和“能够答对但输出不稳定”。

从同一V6起点完成严格／分层奖励GRPO对照，两种奖励各运行两个seed、50次有效更新，以严格任务成功率、协议合法性和严重错误验收。分层奖励未稳定通过旧业务回归门槛，保留严格奖励和原V6，不以训练reward上涨判定改进。调试过程与结果见[测试记录](model/test/test_log/test_grpo_reward_ab_2026-10-30.md)。

