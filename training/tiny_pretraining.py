import wandb
import numpy

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from training.utils.model_checkpoint import ModelCheckpoint
from training.models.tiny_extractor import TinyExtractor
from training.models.tiny_mnist_extractor import TinyMNISTExtractor
from dataset.tiny_cifar10 import TinyCIFAR10DataModule
from dataset.tiny_mnist import TinyMNISTDataModule
from training.utils.utils import log_msg


def tiny_pretraining(config, args, isTune=False):
    args.__dict__.update(config)
    
    if not isTune and args.fast_dev_run == 0:
        wandb_logger = WandbLogger(project='SSL')
    log_msg(args)

    if args.dataset == "cifar10": 
        dm = TinyCIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=args.val_split) 
        args.num_samples = dm.num_samples
        model = TinyExtractor(**args.__dict__)
    elif args.dataset == "mnist":
        dm = TinyMNISTDataModule(data_dir=args.data_dir, train_subset_name=args.train_subset_name, val_subset_name=args.val_subset_name, 
                                 batch_size=args.batch_size, num_workers=args.num_workers)
        model = TinyMNISTExtractor(**args.__dict__)
    else:
        raise NotImplementedError("other datasets have not been implemented till now")
    
    if args.fast_dev_run == 0:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", dirpath=f"results/{wandb.run.name}/")
        callbacks = [model_checkpoint, lr_monitor]
    else:
        callbacks = [] 

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

    if not isTune and args.fast_dev_run == 0:
        trainer_args["logger"] = wandb_logger
    
    trainer = Trainer(**trainer_args)

    trainer.fit(model, datamodule=dm)

    if not isTune and args.fast_dev_run == 0: 
        wandb.finish()