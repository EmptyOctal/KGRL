import torch
import torch.nn as nn
import torch.nn.functional as F

class TransR(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin, p_norm=1):
        super(TransR, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.p_norm = p_norm

        # 初始化实体和关系的嵌入向量
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # 初始化嵌入
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, head, relation, tail):
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)

        # 使用 L1/L2 范数计算距离
        score = torch.norm(head_emb + relation_emb - tail_emb, p=self.p_norm, dim=1)
        return score

    def loss_function(self, positive_score, negative_score):
        # 基于margin的排名损失函数
        return F.relu(positive_score - negative_score + self.margin).mean()
