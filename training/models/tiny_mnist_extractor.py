import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.datasets import TrialCIFAR10

from training.losses.spectral_contrastive_loss import spectral_contrastive_loss
from training.losses.nt_xent_loss import nt_xent_loss
from training.models.projection import Projection

class TinyMNISTExtractor(pl.LightningModule):
    dataset_cls = TrialCIFAR10
    dims = (3, 32, 32)

    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
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

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x28x28
        self.pool = nn.MaxPool2d(2, 2)  # Output: 64x14x14
        self.dropout_conv = nn.Dropout(0.25)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout_fc = nn.Dropout(0.5)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.projection = Projection(input_dim=64, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim, norm_p=self.norm_p, mu=self.projection_mu)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = self.flatten(x)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        return x

    def shared_step(self, batch):
        img1, img2, _ = batch

        h1 = self(img1)
        h2 = self(img2)

        z1 = self.projection(h1)
        z2 = self.projection(h2)

        # TODO change back to spectral_contrastive_loss
        return spectral_contrastive_loss(out_1=z1, out_2=z2)
        #return nt_xent_loss(z1, z2, 0.1)
    
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