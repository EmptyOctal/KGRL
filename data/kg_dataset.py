from torch.utils.data import Dataset

class KGDataset(Dataset):
    def __init__(self, triples, entity2id, relation2id):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        return self.entity2id[head], self.relation2id[relation], self.entity2id[tail]

