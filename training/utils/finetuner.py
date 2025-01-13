from typing import Optional, Tuple

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F 
from torchmetrics import Accuracy

class SSLFineTuner(LightningModule):
    '''
        This is a bug-fixed version of https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/models/self_supervised/ssl_finetuner.py
    '''
    def __init__(
        self,
        backbone: torch.nn.Module,
        in_features: int = 128,
        num_classes: int = 10,
        epochs: int = 25,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = "cosine",
        decay_epochs: Tuple = (60, 80),
        gamma: float = 0.1,
        final_lr: float = 0.0,
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.final_lr = final_lr
        self.dropout = dropout
        self.in_features = in_features
        self.num_classes = num_classes

        self.backbone = backbone
        self.linear_layer = nn.Linear(self.in_features, self.num_classes, bias=True)

        # metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, compute_on_step=False, top_k=1)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, compute_on_step=False, top_k=1)

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(logits.softmax(-1), y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc_step", acc, prog_bar=True)
        self.log("train_acc_epoch", self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.val_acc(logits.softmax(-1), y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.test_acc(logits.softmax(-1), y)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", self.test_acc)

        return loss

    def shared_step(self, batch):
        x, y = batch
        if y.min() < 0 or y.max() >= 10:
            print("Invalid target labels!")
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Input data contains NaN or Inf!")

        with torch.no_grad():
            feats = self.backbone(x)

        feats = feats.view(feats.size(0), -1)

        logits = self.linear_layer(feats)
        if torch.isnan(logits).any():
            print("Logits contain NaN!")
        if torch.isinf(logits).any():
            print("Logits contain Inf!")
        loss = F.cross_entropy(logits, y)
        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.linear_layer.parameters(),
            lr=self.learning_rate,
            nesterov=self.nesterov,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]