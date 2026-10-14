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
│   ├── SOURCE.json
│   ├── pretrain_t2t_mini.jsonl
│   ├── fixtures/            # 小型管线验证数据
│   └── processed/           # 生成的训练/验证划分，Git 忽略
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

检查当前基础模型的性能，进行直观测试和数值测试，保存结果已经保存到路径test/test_log/test_base_model_2026-10-13.md，并且罗列接下来的工作路线

## 10.14

进行基础模型的数值分析收尾，并且正式开始进行SFT训练，自行完成SFT框架