import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

from training.losses.spectral_contrastive_loss import spectral_contrastive_loss
from training.models.projection import Projection
from pl_bolts.datasets import TrialCIFAR10

class TinyExtractor(pl.LightningModule):
    dataset_cls = TrialCIFAR10
    dims = (3, 32, 32)

    def __init__(self, feat_dim=100, learning_rate=1e-3, hidden_dim=128, output_dim=32 , norm_p=2., mu=1., **kwargs):
        super(TinyExtractor, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_p = norm_p
        self.projection_mu = mu


        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 64x32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                  # 64x16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 128x16x16
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # 256x16x16
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feat_dim)

        self.projection = Projection(input_dim=100, hidden_dim=self.hidden_dim, output_dim=self.output_dim, norm_p=self.norm_p, mu=self.projection_mu)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)  
        x = self.fc(x) 
        return x

    def shared_step(self, batch):
        (img1, img2, _), y = batch

        h1 = self(img1)
        h2 = self(img2)

        z1 = self.projection(h1)
        z2 = self.projection(h2)

        return spectral_contrastive_loss(out_1=z1, out_2=z2)
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = self.parameters()
        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        
        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
