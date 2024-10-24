import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, conv_channels, kernel_sizes, dropout, margin=1.0):
        super(SelectE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.margin = margin

        # Embeddings for entities and relations
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Multi-scale convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, conv_channels, (ks, ks), padding=ks//2) for ks in kernel_sizes
        ])

        # Fully connected layers for adaptive selection
        input_dim = conv_channels * len(kernel_sizes)  # 输入维度应为卷积输出的总通道数
        self.fc1 = nn.Linear(input_dim, input_dim // 2)  # 取一半作为隐藏层维度
        self.fc2 = nn.Linear(input_dim // 2, input_dim)  # 输出维度与卷积输出的通道数匹配

        # Output projection layer (adjust input_dim to 19200 if that's the actual size)
        flattened_size = conv_channels * len(kernel_sizes) * self.compute_flattened_size()
        self.fc_out = nn.Linear(flattened_size, embedding_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def compute_flattened_size(self):
        # 假设特征矩阵是 (2, embedding_dim)，经过卷积后的空间尺寸取决于卷积核大小和填充方式
        # 你可以在这里调整空间尺寸，确保计算出的尺寸是准确的
        dummy_input = torch.zeros(1, 1, 2, self.embedding_dim)
        dummy_output = self.convs[0](dummy_input)  # 假设所有卷积层的输出空间尺寸一致
        return dummy_output.size(2) * dummy_output.size(3)

    def forward(self, head, relation, tail):
        # Retrieve embeddings
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)

        # Construct feature matrix with a "chessboard" reshaping
        feature_matrix = torch.cat([head_emb.unsqueeze(1), relation_emb.unsqueeze(1)], dim=1)
        feature_matrix = feature_matrix.view(-1, 1, 2, self.embedding_dim)  # Shape: [batch_size, 1, 2, embedding_dim]

        # Multi-scale convolution
        conv_outputs = [F.relu(conv(feature_matrix)) for conv in self.convs]
        conv_outputs = [self.dropout(output) for output in conv_outputs]
        conv_outputs = torch.cat(conv_outputs, dim=1)  # Concatenate along channel dimension

        # Adaptive feature selection
        pooled = torch.mean(conv_outputs, dim=(2, 3))  # Global average pooling
        adaptive_weights = F.sigmoid(self.fc2(F.relu(self.fc1(pooled))))

        # Adjust feature weighting
        if adaptive_weights.size(1) != conv_outputs.size(1):
            raise ValueError(f"adaptive_weights dimension {adaptive_weights.size(1)} does not match conv_outputs {conv_outputs.size(1)}")

        selected_features = conv_outputs * adaptive_weights.unsqueeze(2).unsqueeze(3)

        # Flatten and project to embedding dimension
        combined_features = selected_features.view(selected_features.size(0), -1)
        projected_emb = self.fc_out(combined_features)

        # Compute score with tail embedding
        score = torch.sum(projected_emb * tail_emb, dim=1)
        return score
    def loss_function(self, positive_score, negative_score):
        # 基于 margin 的排名损失函数
        return torch.sum(F.relu(positive_score - negative_score + self.margin))