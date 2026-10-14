# TrainPilot

本项目是基于minimind的自学习开发项目，目的是从零开始构建一个轻量级语言模型，为用户提供规模训练中的结果分析。本项目在考虑实用性的同时，更多的是模型构建的学习项目，从零开始对模型的训练过程进行拆分重构，并在原有基础上进行改进。本项目中可能涉及到学习过程的记录，可酌情参考。

## 项目结构如下

```text
TrainPilot/
├── config/                  # 配置类、模型与训练 JSON 配置
│   ├── config.py
│   ├── pretrain.json
│   └── smoke.json
├── dataset/                 # 数据预处理、分词器、语料与来源说明
│   ├── dataset.py
│   ├── tokenizer.json
│   ├── tokenizer_config.json # SFT 聊天模板与特殊 token 配置
│   ├── SOURCE.json
│   ├── datasets/            # 语料与样例统一存放
│   │   ├── pretrain_t2t_mini.jsonl
│   │   ├── sft_t2t_mini.jsonl
│   │   ├── fixtures/        # 小型管线验证数据
│   │   └── processed/       # 生成的训练/验证划分，Git 忽略
│   └── SFT_SOURCE.json      # SFT 数据来源与校验记录
├── images/                  # 图片目录，待补充
├── model/                   # 模型与基础层
│   ├── layers.py
│   ├── model.py
│   └── minimind.LICENSE
├── trainer/                 # 训练函数、epoch/验证/保存、启动入口
│   ├── pretrain.py
│   ├── trainer.py
│   └── train.py
├── test/
│   ├── test_code/           # 单元检查、交互测试与实验复现脚本
│   └── test_log/            # 测试日志与数值结果 JSON
├── runs/                    # 本地权重与运行产物，Git 忽略
├── requirements.txt
├── THIRD_PARTY_NOTICES.md
└── README.md
```

重排后的命令（在仓库根目录执行）：

```bash
python -m trainer.train --config config/smoke.json
python -m trainer.train --config config/pretrain.json
python -m test.test_code.base_model_test --checkpoint runs/pre_train1-20261012/checkpoint.pt
python -m unittest discover -s test/test_code -p "test_*.py" -v
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