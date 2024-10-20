import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from .kg_dataset import KGDataset, load_data

class DInterface(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=128):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self, stage=None):
        triples, entity2id, relation2id = load_data(self.data_path)
        self.dataset = KGDataset(triples, entity2id, relation2id)
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.num_entities = len(entity2id)
        self.num_relations = len(relation2id)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)