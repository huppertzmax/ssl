import wandb
import numpy

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
from torchvision import transforms

from training.models.ssl_online import SSLOnlineEvaluator
from training.utils.model_checkpoint import ModelCheckpoint
from training.models.simclr import SimCLR
from training.utils.utils import log_msg


def pretraining(config, args, isTune=False):
    args.__dict__.update(config)
    
    if not isTune:
        wandb_logger = WandbLogger(project='SSL')
    log_msg(args)

    if args.norm_p == -1.:
        args.norm_p = numpy.inf
    if args.distance_p == -1.:
        args.distance_p = numpy.inf

    if args.dataset == "cifar10" or args.dataset == "cifar100": #TODO cifar100 implementation
        dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=args.val_split) 
        args.num_samples = dm.num_samples
        normalization = cifar10_normalization()
    elif args.dataset == "mnist":
        dm = MNISTDataModule(data_dir=args.data_dir, normalize=False, batch_size=args.batch_size, num_workers=args.num_workers, val_split=args.val_split)
        normalization = transforms.Normalize((0.1307,), (0.3081,))
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=dm.dims[-1],
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=dm.dims[-1],
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    model = SimCLR(**args.__dict__)

    online_evaluator = None
    if isTune | args.online_ft:
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=args.hidden_mlp,
            num_classes=dm.num_classes,
            dataset=args.dataset,
            isTune=isTune,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", dirpath=f"results/{wandb.run.name}/")
    callbacks = [] if isTune else [model_checkpoint]
    if isTune | args.online_ft:
        callbacks.append(online_evaluator)
    callbacks.append(lr_monitor)

    trainer_args = {
        "max_epochs": args.max_epochs,
        "devices": args.gpus,
        "num_nodes": args.num_nodes,
        "accelerator": "gpu",
        "sync_batchnorm": True if args.gpus > 1 else False,
        "precision": 32 if args.fp32 else 16,
        "callbacks": callbacks,
        "fast_dev_run": args.fast_dev_run,
    }
    
    if not isTune:
        trainer_args["logger"] = wandb_logger

    trainer = Trainer(**trainer_args)

    trainer.fit(model, datamodule=dm)

    if not isTune: 
        wandb.finish()