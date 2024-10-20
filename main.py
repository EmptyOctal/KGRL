# main.py
import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from data.data_interface import DInterface
from models.model_interface import MInterface

def train(args):
    data_module = DInterface(data_path=args.data_path, batch_size=args.batch_size)
    data_module.setup(stage='fit')  # 确保数据模块已设置好

    model = MInterface(num_entities=data_module.num_entities,
                       num_relations=data_module.num_relations,
                       embedding_dim=args.embedding_dim,
                       margin=args.margin,
                       lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
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
    test_loader = data_module.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch)
            print(preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/subgraph_kgp1.txt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--is_train', action='store_true')

    args = parser.parse_args()
    train(args)
    # if args.is_train:
    #     train(args)
    # else:
    #     predict(args)