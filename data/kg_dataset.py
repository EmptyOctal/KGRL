import pandas as pd
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

def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, usecols=range(8))
    data.columns = ['id', 'orgin_id', 'start_lang', 'end_lang', 'weight', 'start_entity', 'relation', 'end_entity']

    triples = data[['start_entity', 'relation', 'end_entity']].values
    entities = set(data['start_entity']).union(set(data['end_entity']))
    relations = set(data['relation'])

    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {relation: idx for idx, relation in enumerate(relations)}

    return triples, entity2id, relation2id