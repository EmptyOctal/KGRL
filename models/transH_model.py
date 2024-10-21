import torch
import torch.nn as nn
import torch.nn.functional as F

class TransH(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin, p_norm=1):
        super(TransH, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.p_norm = p_norm

        # 初始化实体、关系和法向量的嵌入
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.normal_vector_embeddings = nn.Embedding(num_relations, embedding_dim)

        # 初始化嵌入
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.normal_vector_embeddings.weight)

    def forward(self, head, relation, tail):
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        normal_vector = self.normal_vector_embeddings(relation)

        # 投影到超平面
        head_proj = self.project_to_hyperplane(head_emb, normal_vector)
        tail_proj = self.project_to_hyperplane(tail_emb, normal_vector)

        # 使用 L1/L2 范数计算得分
        score = torch.norm(head_proj + relation_emb - tail_proj, p=self.p_norm, dim=1)
        return score

    def project_to_hyperplane(self, entity_emb, normal_vector):
        # 将实体投影到法向量定义的超平面上
        return entity_emb - torch.sum(entity_emb * normal_vector, dim=1, keepdim=True) * normal_vector

    def loss_function(self, positive_score, negative_score):
        # 基于 margin 的排名损失函数
        return torch.sum(F.relu(positive_score - negative_score + self.margin))
