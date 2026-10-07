import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy


class MultiLayerPerceptron(pl.LightningModule):
    """
    Multi-Layer Perceptron (MLP) with PyTorch Lightning for image classification,
    specifically for MNIST dataset.
    """

    def __init__(self, input_size=(1, 28, 28), hidden_size=(32, 16), output_size=10):
        super().__init__()

        self.train_acc = Accuracy(task="multiclass", num_classes=output_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=output_size)
        self.test_acc = Accuracy(task="multiclass", num_classes=output_size)

        flat_input_size = input_size[0] * input_size[1] * input_size[2]

        layers = [nn.Flatten()]
        for hidden_unit in hidden_size:
            layers.append(nn.Linear(flat_input_size, hidden_unit))
            layers.append(nn.ReLU())
            flat_input_size = hidden_unit
        layers.append(nn.Linear(flat_input_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        self.test_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
