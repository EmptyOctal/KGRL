import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from .kg_dataset import KGDataset
from utils import load_train_data, load_test_data

class DInterface(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=128, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self, stage=None):
            # 加载训练集
            train_triples, train_entity2id, train_relation2id = load_train_data(self.data_path)
            self.train_dataset = KGDataset(train_triples, train_entity2id, train_relation2id)
            self.entity2id = train_entity2id
            self.relation2id = train_relation2id
            self.num_entities = len(train_entity2id)
            self.num_relations = len(train_relation2id)
            # 加载验证集和测试集
            test_triples, test_entity2id, test_relation2id = load_test_data(self.data_path)
            self.val_dataset = KGDataset(test_triples, test_entity2id, test_relation2id)
            self.test_dataset = KGDataset(test_triples, test_entity2id, test_relation2id)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
