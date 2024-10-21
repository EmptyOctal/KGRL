import torch
import torch.nn as nn
import torch.nn.functional as F

class TransR(nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, relation_dim, margin, p_norm=1):
        super(TransR, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_dim = entity_dim  # 实体空间
        self.relation_dim = relation_dim  # 关系空间
        self.margin = margin
        self.p_norm = p_norm

        # 初始化实体和关系的嵌入向量
        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)
        # 初始化关系映射矩阵
        self.relation_transfer = nn.Embedding(num_relations, entity_dim * relation_dim)

        # 初始化嵌入
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_transfer.weight)

    def forward(self, head, relation, tail):
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        relation_matrix = self.relation_transfer(relation)

        # 将头实体和尾实体嵌入投影到关系空间
        head_proj = self.project_to_relation_space(head_emb, relation_matrix)
        tail_proj = self.project_to_relation_space(tail_emb, relation_matrix)

        # 使用 L1/L2 范数计算得分
        score = torch.norm(head_proj + relation_emb - tail_proj, p=self.p_norm, dim=1)
        return score

    def project_to_relation_space(self, entity_emb, relation_matrix):
        # 将实体嵌入投影到关系空间，形状转换
        entity_emb = entity_emb.unsqueeze(1)  # [batch_size, 1, entity_dim]
        relation_matrix = relation_matrix.view(-1, self.entity_dim, self.relation_dim)  # [batch_size, entity_dim, relation_dim]
        projected_emb = torch.matmul(entity_emb, relation_matrix).squeeze(1)  # [batch_size, relation_dim]
        return projected_emb

    def loss_function(self, positive_score, negative_score):
        # 基于 margin 的排名损失函数
        torch.sum(F.relu(positive_score - negative_score + self.margin))
