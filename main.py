import os
import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from data.data_interface import DInterface
from models.model_interface import MInterface
from utils.process import process_data

def train(args):
    data_dir = "dataset/processed/"
    data_module = DInterface(data_path=data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    data_module.setup(stage='fit')

    model = MInterface(num_entities=data_module.num_entities,
                       num_relations=data_module.num_relations,
                       embedding_dim=args.embedding_dim,
                       margin=args.margin,
                       lr=args.lr,
                       model_name=args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    # logger = CSVLogger("logs", name="transE")
    logger = WandbLogger(project="KGRL", name="transE")
    trainer = Trainer(max_epochs=args.max_epochs, 
                        callbacks=[checkpoint_callback, early_stopping_callback], 
                        logger=logger)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/raw/subgraph_kgp1.txt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--model_checkpoint', type=str, default='logs/transE/version_15/checkpoints/epoch=0-step=3907.ckpt')
    parser.add_argument('--valid_json', type=str, default='dataset/subgraph_kgp1_valid.json')
    parser.add_argument('--output_json', type=str, default='dataset/subgraph_kgp1_output.json')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--model_name', type=str, default='transE', choices=['transE', 'transR'])

    args = parser.parse_args()
    # 加工数据
    processed_dir = 'dataset/processed/'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        process_data(args.data_path, processed_dir)
    # 训练
    train(args)
    # if args.is_train:
    #     train(args)
    # else:
    #     predict(args)