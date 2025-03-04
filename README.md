# KGRL: 知识图谱表示学习

KGRL 是一个基于 PyTorch Lightning 的知识图谱表示学习框架，支持 TransE、TransH、TransR、SelectE 等多种模型。本项目提供了模块化的结构，便于研究不同的知识图谱嵌入方法，并进行链路预测（Link Prediction）和实体预测（Entity Prediction）等任务。

## 项目优势
- **模块化设计**：项目结构清晰，方便扩展和实验不同的嵌入方法以及使用不同的数据集。
- **支持模型**：目前支持 TransE、TransH、TransR、SelectE 四种经典模型，易于添加新的模型。
- **任务支持**：包括链路预测和实体预测任务。


## 项目结构：
```bash
KGRL/
├── data/   # 数据集处理相关代码
│   ├── __init__.py
│   ├── data_interface.py # 数据接口类，用于数据的加载与预处理
│   └── kg_dataset.py
├── dataset/
│   ├── processed/ # 加工后的数据存放位置
│   ├──── entity2id.txt
│   ├──── relation2id.txt
│   ├──── test.txt
│   ├──── train.txt
│   ├── raw/       # 原始数据存放位置
│   ├──── subgraph_kgp1_output.json
│   ├──── subgraph_kgp1_valid.json
│   └── test1_demo.txt
├── eval_script.py # 评测代码
├── models/   # 模型相关代码
│   ├── __init__.py
│   ├── model_interface.py
│   ├── selectE_model.py
│   ├── transE_model.py
│   ├── transH_model.py
│   └── transR_model.py
├── utils/ # 工具函数和辅助脚本
│   ├── __init__.py
│   ├── load.py  # 加载数据
│   └── process.py # 提取数据
├── main.py
├── requirements.txt # 项目环境依赖
└── README.md
```

## 使用方法

### 1. 克隆项目
首先，使用以下命令将项目克隆到本地：
```bash
git clone git@github.com:EmptyOctal/KGRL.git
cd KGRL
```

### 2. 环境依赖
首先，安装所需的Python依赖。可以通过以下命令安装：

```bash
pip install -r requirements.txt
```
或自行使用包管理工具安装如下包：
- torch>=2.0
- pytorch-lightning>=2.0
- numpy>=1.21
- pandas>=1.3
- tqdm>=4.62

### 3. 数据准备
将知识图谱数据文件放置于 data/raw/ 文件夹中。本项目支持以 .txt 格式存储的三元组数据，示例数据为subgraph_kgp1.txt，数据集传送门：[GoogleDrive下载链接](https://drive.google.com/drive/folders/1sN04rVzAzysszhWvG-njMQmS0EINSXGW?usp=sharing)（该文件为课程项目所用数据集，源自于[DBpedia知识库](https://www.dbpedia.org/)，属于RDF格式的三元组数据）。
运行代码后，会自动在 data/processed/ 目录下生成经过提取和数据增强后形成的txt文件，新生成的文件每一行表示一个`实体-关系-实体`的三元组关系。

### 4. 训练模型
使用以下命令训练模型：
```bash
python main.py --mode train --model TransE --data_path ./data/subgraph_kgp1.txt --epochs 100
```

### 5. 模型测试
训练完成后，使用以下命令进行测试和预测：
```bash
python main.py --mode predict --model_checkpoint /path/to/model_checkpoint/
```
测试过程为对 `subgraph_kgp1_valid.json` 的数据进行链路预测（Link Prediction）和实体预测（Entity Prediction），覆盖output的内容后输出为 `subgraph_kgp1_output.json`。

可以再在根目录执行，进行评分
```bash
python eval_script.py
```

### 6. demo 演示
可执行以下命令：
```bash
python main.py --mode predict_demo --model_checkpoint /path/to/model_checkpoint/
```
运行后，可以手动输入内容进行测试：
```bash
请输入头实体: xxx
请输入尾实体或关系: xxx
预测的关系前五名是:
...
```

## 扩展与自定义
你可以根据需要扩展现有模型或加入新的模型。只需在 models/ 文件夹中实现新模型，并在 main.py 中注册即可。

## 项目参数解释
- data_path: 数据文件路径，默认值为 dataset/raw/subgraph_kgp1.txt
- batch_size: 批处理大小，默认值为 512
- embedding_dim: 嵌入维度，默认值为 100
- entity_dim: 实体嵌入维度，默认值为 100
- relation_dim: 关系嵌入维度，默认值为 50
- margin: 损失函数中的边距，默认值为 1.0
- lr: 学习率，默认值为 0.0001
- max_epochs: 最大训练轮数，默认值为 100
- model_checkpoint: 模型检查点路径，默认值为 None
- valid_json: 验证集 JSON 文件路径，默认值为 None
- output_json: 输出结果 JSON 文件路径，默认值为 None
- num_workers: 工作线程数，默认值为 16
- mode: 选择模式训练还是预测，默认值为 train, 可选值为 predict 或 predict_demo
- model_name: 模型名称，默认值为 transE，目前已实现的可选值为 transH 或 transR