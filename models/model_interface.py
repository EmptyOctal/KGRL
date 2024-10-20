from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
from .transE_model import TransEModel

class MInterface(LightningModule):
    def __init__(self, num_entities, num_relations, embedding_dim, margin, lr):
        super(MInterface, self).__init__()
        self.model = TransEModel(num_entities, num_relations, embedding_dim, margin)
        self.lr = lr
        self.num_entities = num_entities

    def forward(self, head, relation, tail):
        return self.model(head, relation, tail)

    def training_step(self, batch, batch_idx):
        head, relation, tail = batch
        if torch.rand(1).item() > 0.5:
            while True:
                neg_head = torch.randint(0, self.num_entities, head.shape, device=self.device)
                if not torch.equal(head, neg_head):
                    break
            neg_tail = tail
        else:
            neg_head = head
            while True:
                neg_tail = torch.randint(0, self.num_entities, tail.shape, device=self.device)
                if not torch.equal(tail, neg_tail):
                    break

        pos_score = self.model(head, relation, tail)
        neg_score = self.model(neg_head, relation, neg_tail)

        loss = self.model.loss_function(pos_score, neg_score)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        head, relation, tail = batch
        pos_score = self.model(head, relation, tail)
        self.log('val_loss', pos_score.mean(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
