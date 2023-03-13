
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class SimpleClassifier(pl.LightningModule):

    def __init__(self,
                 input_neurons,
                 dropout=0,
                 n1=500,
                 n2=250,
                 lr=0.001):
        super().__init__()

        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(task='binary')
        self.auroc = torchmetrics.AUROC(task='binary')

        # Define layers
        self.linear1 = nn.Linear(input_neurons, n1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(n1, n2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(n2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, acc, auroc = self._shared_eval_step(batch, batch_idx)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('train_auroc', auroc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, auroc = self._shared_eval_step(batch, batch_idx)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_auroc', auroc)

    def _shared_eval_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.acc(y_hat, y)
        auroc = self.auroc(y_hat, y)
        return loss, acc, auroc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)