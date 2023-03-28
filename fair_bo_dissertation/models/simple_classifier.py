
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
        self.loss = nn.BCELoss()
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

        # Metrics
        self.confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix()

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
        x, y, _ = batch
        y_hat = self(x).squeeze(-1)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def on_validation_start(self):
        self.validation_confmat = torch.zeros((2, 2))
        self.validation_protected_confmats = {}

    def validation_step(self, batch, batch_idx):
        x, y, x_protected = batch
        y_hat = self(x).squeeze(-1)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        self.validation_confmat += self.confusion_matrix(y_hat, y)
        for x_protected_value in torch.unique(x_protected):
            x_protected_value = x_protected_value.item()
            y_p = y[x_protected == x_protected_value]
            y_hat_p = y_hat[x_protected == x_protected_value]
            if x_protected_value not in self.validation_protected_confmats.keys():
                self.validation_protected_confmats[x_protected_value] = torch.zeros((2,2))
            self.validation_protected_confmats[x_protected_value] += self.confusion_matrix(y_hat_p, y_p)

    def on_validation_end(self):
        print(self.validation_confmat)
        print(self.validation_protected_confmats)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)