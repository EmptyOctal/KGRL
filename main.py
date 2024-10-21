import os
import argparse
import torch
import json
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from data.data_interface import DInterface
from models.model_interface import MInterface
from utils.process import process_data, aug_data
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import wandb

def train(args):
    if args.augment:
        data_dir = 'dataset/augmented/'
    else:
        data_dir = 'dataset/processed/'
    data_module = DInterface(data_path=data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    data_module.setup(stage='fit')

    model = MInterface(num_entities=data_module.num_entities,
                       num_relations=data_module.num_relations,
                       embedding_dim=args.embedding_dim,
                       entity_dim=args.entity_dim,
                       relation_dim=args.relation_dim,
                       margin=args.margin,
                       lr=args.lr,
                       model_name=args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    # logger = CSVLogger("logs", name="transE")
    wandb.init(project='KGRL', config=args)
    # 创建 WandbLogger
    wandb_logger = WandbLogger()
    trainer = Trainer(max_epochs=args.max_epochs, 
                        callbacks=[checkpoint_callback], 
                        logger=wandb_logger)
    trainer.fit(model, data_module)

def predict_demo(args):
    data_dir = "dataset/processed/"
    data_module = DInterface(data_path=data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    data_module.setup(stage='test')

    model = MInterface.load_from_checkpoint(args.model_checkpoint,
                                            num_entities=data_module.num_entities,
                                            num_relations=data_module.num_relations,
                                            embedding_dim=args.embedding_dim,
                                            entity_dim=args.entity_dim,
                                            relation_dim=args.relation_dim,
                                            margin=args.margin,
                                            lr=args.lr,
                                            model_name=args.model_name)
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    entity2id = data_module.entity2id
    relation2id = data_module.relation2id
    id2entity = {v: k for k, v in entity2id.items()}

    head = input("请输入头实体: ")
    tail_or_relation = input("请输入尾实体或关系: ")

    if tail_or_relation in entity2id:
        # 输入的是尾实体，预测关系
        head_id = entity2id[head]
        tail_id = entity2id[tail_or_relation]
        head_tensor = torch.tensor([head_id], dtype=torch.long).to(model.device)
        tail_tensor = torch.tensor([tail_id], dtype=torch.long).to(model.device)

        scores = []
        for relation, relation_id in relation2id.items():
            relation_tensor = torch.tensor([relation_id], dtype=torch.long).to(model.device)
            score = model(head_tensor, relation_tensor, tail_tensor).item()
            scores.append((relation, score))
        
        # 按分数排序并取前五个
        top_5_relations = sorted(scores, key=lambda x: x[1])[:5]
        print("预测的关系前五名是:")
        for relation, score in top_5_relations:
            print(f"{relation}: {score}")

    elif tail_or_relation in relation2id:
        # 输入的是关系，预测尾实体
        head_id = entity2id[head]
        relation_id = relation2id[tail_or_relation]
        head_tensor = torch.tensor([head_id], dtype=torch.long).to(model.device)
        relation_tensor = torch.tensor([relation_id], dtype=torch.long).to(model.device)

        # 批量处理实体评分
        entity_ids = list(entity2id.values())
        entity_tensors = torch.tensor(entity_ids, dtype=torch.long).to(model.device)

        batch_size = 1024  # 设置批次大小
        scores = []
        for i in range(0, len(entity_ids), batch_size):
            batch_tail_tensors = entity_tensors[i:i+batch_size]
            batch_head_tensors = head_tensor.repeat(batch_tail_tensors.size(0))
            batch_relation_tensors = relation_tensor.repeat(batch_tail_tensors.size(0))

            # 批量预测
            batch_scores = model(batch_head_tensors, batch_relation_tensors, batch_tail_tensors).tolist()
            scores.extend(zip(entity_ids[i:i+batch_size], batch_scores))

        # 按分数排序并取前五个
        top_5_entities = sorted(scores, key=lambda x: x[1])[:5]
        print("预测的尾实体前五名是:")
        for entity_id, score in top_5_entities:
            print(f"{id2entity[entity_id]}: {score}")

    else:
        print("输入有误，请输入有效的尾实体或关系")


def predict(args):
    # 加载数据模块和模型
    data_dir = "dataset/processed/"
    data_module = DInterface(data_path=data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    data_module.setup(stage='test')

    model = MInterface.load_from_checkpoint(args.model_checkpoint,
                                            num_entities=data_module.num_entities,
                                            num_relations=data_module.num_relations,
                                            embedding_dim=args.embedding_dim,
                                            entity_dim=args.entity_dim,
                                            relation_dim=args.relation_dim,
                                            margin=args.margin,
                                            lr=args.lr,
                                            model_name=args.model_name)
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    entity2id = data_module.entity2id
    relation2id = data_module.relation2id
    id2entity = {v: k for k, v in entity2id.items()}

    # 读取 valid_json 文件
    with open(args.valid_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理 link_prediction 任务
    print("Processing link prediction tasks...")
    for task in tqdm(data["link_prediction"], desc="Link Prediction"):
        head, tail = task["input"]
        if head in entity2id and tail in entity2id:
            head_id = entity2id[head]
            tail_id = entity2id[tail]
            head_tensor = torch.tensor([head_id], dtype=torch.long).to(model.device)
            tail_tensor = torch.tensor([tail_id], dtype=torch.long).to(model.device)

            scores = []
            for relation, relation_id in relation2id.items():
                relation_tensor = torch.tensor([relation_id], dtype=torch.long).to(model.device)
                score = model(head_tensor, relation_tensor, tail_tensor).item()
                scores.append((relation, score))

            # 按分数排序并取前五个
            top_5_relations = sorted(scores, key=lambda x: x[1])[:5]
            task["output"] = [relation for relation, _ in top_5_relations]

    # 处理 entity_prediction 任务，使用批量计算
    print("Processing entity prediction tasks...")
    def process_entity_prediction_batch(task, batch_size=1024):
        head, relation = task["input"]
        if head in entity2id and relation in relation2id:
            head_id = entity2id[head]
            relation_id = relation2id[relation]
            head_tensor = torch.tensor([head_id], dtype=torch.long).to(model.device)
            relation_tensor = torch.tensor([relation_id], dtype=torch.long).to(model.device)

            # 批量处理实体评分
            entity_ids = list(entity2id.values())
            entity_tensors = torch.tensor(entity_ids, dtype=torch.long).to(model.device)

            # 切分为批次
            scores = []
            for i in range(0, len(entity_ids), batch_size):
                batch_tail_tensors = entity_tensors[i:i+batch_size]
                batch_head_tensors = head_tensor.repeat(batch_tail_tensors.size(0))
                batch_relation_tensors = relation_tensor.repeat(batch_tail_tensors.size(0))

                # 批量预测
                batch_scores = model(batch_head_tensors, batch_relation_tensors, batch_tail_tensors).tolist()
                scores.extend(zip(entity_ids[i:i+batch_size], batch_scores))

            # 按分数排序并取前五个
            top_5_entities = sorted(scores, key=lambda x: x[1])[:5]
            task["output"] = [id2entity[entity_id] for entity_id, _ in top_5_entities]
        return task

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # 使用多线程 + 批量计算的方式执行 entity_prediction 部分
        results = list(tqdm(executor.map(process_entity_prediction_batch, data["entity_prediction"]),
                            total=len(data["entity_prediction"]),
                            desc="Entity Prediction"))

    # 更新 entity_prediction 的结果
    data["entity_prediction"] = results

    # 将结果写入 output_json 文件
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("预测结果已保存到", args.output_json)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/raw/subgraph_kgp1.txt')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--entity_dim', type=int, default=100) # 实体空间
    parser.add_argument('--relation_dim', type=int, default=50) # 关系空间
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--model_checkpoint', type=str, default='lightning_logs/6lf63jco/checkpoints/epoch=35-step=42840.ckpt')
    parser.add_argument('--valid_json', type=str, default='dataset/subgraph_kgp1_valid.json')
    parser.add_argument('--output_json', type=str, default='dataset/subgraph_kgp1_output.json')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'predict_demo'])
    parser.add_argument('--model_name', type=str, default='transH', choices=['transE', 'transR'])
    parser.add_argument('--augment', type=bool, default=False)

    args = parser.parse_args()
    # 加工数据
    processed_dir = 'dataset/processed/'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        process_data(args.data_path, processed_dir)
    
    if args.augment:
        aug_data_dir = 'dataset/augmented/'
        if not os.path.exists(aug_data_dir):
            os.makedirs(aug_data_dir)
            aug_data(args.data_path, aug_data_dir)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
    elif args.mode == 'predict_demo':
        while True:
            predict_demo(args)
