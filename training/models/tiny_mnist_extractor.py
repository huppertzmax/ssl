import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.datasets import TrialCIFAR10

from training.losses.spectral_contrastive_loss import spectral_contrastive_loss
from training.models.projection import Projection
from training.models.tiny_mnist_backbone import TinyMNISTBackbone

class TinyMNISTExtractor(pl.LightningModule):
    def __init__(
        self,
        gpus: int= 1,
        num_samples: int = 1024,
        batch_size: int = 256,
        dataset: str ="mnist",
        num_nodes: int = 1,
        arch: str = "custom architecture",
        hidden_mlp: int = 128,
        feat_dim: int = 32,
        warmup_epochs: int = 10,
        max_epochs: int = 50,
        optimizer: str = "adam",
        start_lr: float = 0.0,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        loss_type: str = "spectral_contrastive_loss",
        projection_mu: float=1.0,
        norm_p: float=2.0,
        use_lr_scheduler: bool=False,
        **kwargs
    ):
        super(TinyMNISTExtractor, self).__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.loss_type = loss_type
        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim 
        self.max_epochs = max_epochs
        self.norm_p = norm_p
        self.projection_mu = projection_mu

        self.optim = optimizer
        self.use_lr_scheduler = use_lr_scheduler

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay

        self.train_iters_per_epoch = self.num_samples // self.batch_size

        self.backbone = TinyMNISTBackbone()

        self.projection = Projection(input_dim=64, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim, norm_p=self.norm_p, mu=self.projection_mu)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def shared_step(self, batch):
        img1, img2, _ = batch

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
        
        if self.use_lr_scheduler:
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
        else:
            return optimizer