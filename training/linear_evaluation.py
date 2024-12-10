import wandb
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision import transforms

from training.models.simclr import SimCLR
from training.utils.model_checkpoint import ModelCheckpoint
from training.utils.transforms import SimCLRFinetuneTransform
from training.utils.utils import log_msg

def linear_evaluation(config, args):
    args.__dict__.update(config)
    seed_everything(1234)
    
    wandb_logger = WandbLogger(project='SSL')
    log_msg(args)

    if args.dataset == "cifar10":
        dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers) 
        normalization = cifar10_normalization()
    elif args.dataset == "mnist":
        dm = MNISTDataModule(data_dir=args.data_dir, normalize=False, batch_size=args.batch_size, num_workers=args.num_workers)
        normalization = transforms.Normalize((0.1307,), (0.3081,))
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SimCLRFinetuneTransform(normalize=normalization, input_height=dm.dims[-1], eval_transform=False)
    dm.val_transforms = SimCLRFinetuneTransform(normalize=normalization, input_height=dm.dims[-1], eval_transform=True)
    dm.test_transforms = SimCLRFinetuneTransform(normalize=normalization, input_height=dm.dims[-1], eval_transform=True)

    backbone = SimCLR(
        gpus=args.gpus,
        nodes=1,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        maxpool1=False,
        first_conv=False,
        dataset=args.dataset,
        feat_dim=args.feat_dim
    ).load_from_checkpoint(args.ckpt_path, strict=False)

    tuner = SSLFineTuner( 
        backbone,
        in_features=args.in_features,
        num_classes=dm.num_classes,
        epochs=args.num_epochs,
        hidden_dim=None,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        scheduler_type=args.scheduler_type,
        gamma=args.gamma,
        final_lr=args.final_lr,
    )

    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", dirpath=f"results/eval/{wandb.run.name}/")
    
    trainer = Trainer(
        devices=args.gpus,
        num_nodes=1,
        precision=32 if args.fp32 else 16,
        max_epochs=args.num_epochs,
        accelerator=args.accelerator,
        sync_batchnorm=True if args.gpus > 1 else False,
        fast_dev_run=args.fast_dev_run,
        logger=wandb_logger,
        callbacks=[model_checkpoint],
    )

    trainer.fit(tuner, dm)
    trainer.test(datamodule=dm)

wandb.finish()