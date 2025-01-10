import os
import torch
import wandb
from torch import optim, nn
from argparse import ArgumentParser
from torchvision import transforms
from torchsummary import summary
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.datamodules import MNISTDataModule
from training.utils.utils import log_msg

from training.models.tiny_mnist_backbone import TinyMNISTBackbone


class MNISTModel(LightningModule):
    def __init__(self, backbone, linear_head):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head
        self.train_acc = Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.val_acc = Accuracy(task="multiclass", num_classes=10, compute_on_step=False, top_k=1)
        self.test_acc = Accuracy(task="multiclass", num_classes=10, compute_on_step=False, top_k=1)
        summary(backbone.cuda(), (1, 28, 28))

    def on_train_epoch_start(self):
        self.backbone.train()
        self.linear_head.train()

    def on_validation_start(self):
        self.backbone.eval()
        self.linear_head.eval()
    
    def on_test_start(self):
        self.backbone.eval()
        self.linear_head.eval()
    
    def step(self, batch):
        x, y = batch
        z = self.backbone(x)
        logits = self.linear_head(z)
        loss = nn.functional.cross_entropy(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        acc = self.train_acc(logits.softmax(-1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc_step", acc, prog_bar=True)
        self.log("train_acc_epoch", self.train_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        self.val_acc(logits.softmax(-1), y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        self.test_acc(logits.softmax(-1), y)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", self.test_acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def supervised_learning(config, args):
    args.__dict__.update(config)
    seed_everything(1234)
    
    if args.fast_dev_run == 0:
        wandb_logger = WandbLogger(project='supervised')
    log_msg(args)

    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dm = MNISTDataModule(data_dir="./dataset", normalize=False, batch_size=args.batch_size, num_workers=4, shuffle=True, val_split=0.15)
    dm.train_transforms = transformation
    dm.val_transforms = transformation
    dm.test_transforms = transformation

    backbone = TinyMNISTBackbone()
    linear_head = nn.Linear(32, 10, bias=True)
    model = MNISTModel(backbone, linear_head)

    trainer_args = {
        "devices": 1,
        "precision": 32,
        "max_epochs": args.num_epochs,
        "accelerator": "gpu",
        "sync_batchnorm": False,
        "fast_dev_run": args.fast_dev_run,
    }

    if args.fast_dev_run == 0:
        trainer_args["logger"] = wandb_logger
    
    trainer = Trainer(**trainer_args)
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

    if args.fast_dev_run == 0:
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_epochs", default=15, type=int, help="number of total epochs to run")
    parser.add_argument("--batch_size", default=512, type=int, help="batch_size")
    parser.add_argument("--fast_dev_run", default=0, type=int)
    args = parser.parse_args()
    supervised_learning({}, args)