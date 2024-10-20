import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from data.data_interface import DInterface
from models.model_interface import MInterface
import json

def train(args):
    data_module = DInterface(data_path=args.data_path, batch_size=args.batch_size)
    data_module.setup(stage='fit')  # 确保数据模块已设置好

    model = MInterface(num_entities=data_module.num_entities,
                       num_relations=data_module.num_relations,
                       embedding_dim=args.embedding_dim,
                       margin=args.margin,
                       lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 检查点保存和日志记录
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    logger = CSVLogger("logs", name="transE")

    # 训练器
    trainer = Trainer(max_epochs=args.max_epochs, 
                        callbacks=[checkpoint_callback], 
                        logger=logger)
    trainer.fit(model, data_module)

def predict(args):
    data_module = DInterface(data_path=args.data_path, batch_size=args.batch_size)
    data_module.setup(stage='test')

    model = MInterface.load_from_checkpoint(args.model_checkpoint,
                                            num_entities=data_module.num_entities,
                                            num_relations=data_module.num_relations,
                                            embedding_dim=args.embedding_dim,
                                            margin=args.margin,
                                            lr=args.lr)

    model.eval()
    entity2id = data_module.entity2id
    relation2id = data_module.relation2id
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    def predict_link(head, tail):
        head_id = entity2id.get(head, None)
        tail_id = entity2id.get(tail, None)
        if head_id is None or tail_id is None:
            return []

        head_tensor = torch.tensor([head_id], device=model.device)
        tail_tensor = torch.tensor([tail_id], device=model.device)

        scores = []
        for relation_id in range(model.model.num_relations):
            relation_tensor = torch.tensor([relation_id], device=model.device)
            score = model(head_tensor, relation_tensor, tail_tensor).item()
            scores.append((score, relation_id))

        scores.sort(reverse=True)
        top_relations = [id2relation[relation_id] for _, relation_id in scores[:5]]
        return top_relations

    def predict_entity(head, relation):
        head_id = entity2id.get(head, None)
        relation_id = relation2id.get(relation, None)
        if head_id is None or relation_id is None:
            return []

        head_tensor = torch.tensor([head_id], device=model.device)
        relation_tensor = torch.tensor([relation_id], device=model.device)

        scores = []
        for tail_id in range(model.model.num_entities):
            tail_tensor = torch.tensor([tail_id], device=model.device)
            score = model(head_tensor, relation_tensor, tail_tensor).item()
            scores.append((score, tail_id))

        scores.sort(reverse=True)
        top_entities = [id2entity[tail_id] for _, tail_id in scores[:5]]
        return top_entities

    with open(args.valid_json, 'r') as f:
        data = json.load(f)

    for item in data['link_prediction']:
        head = item['input'][0]
        tail = item['output'][0]
        item['output'] = predict_link(head, tail)

    for item in data['entity_prediction']:
        head = item['input'][0]
        relation = item['output'][0]
        item['output'] = predict_entity(head, relation)

    with open(args.output_json, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/subgraph_kgp1.txt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--valid_json', type=str, default='dataset/subgraph_kgp1_valid.json')
    parser.add_argument('--output_json', type=str, default='dataset/subgraph_kgp1_output.json')
    parser.add_argument('--is_train', action='store_true')

    args = parser.parse_args()
    train(args)
    # if args.is_train:
    #     train(args)
    # else:
    #     predict(args)