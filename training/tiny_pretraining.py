import wandb
import numpy

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform

from training.utils.model_checkpoint import ModelCheckpoint
from training.models.tiny_extractor import TinyExtractor
from tiny_cifar10 import TinyCIFAR10DataModule
from training.utils.utils import log_msg


def tiny_pretraining(config, args, isTune=False):
    args.__dict__.update(config)
    
    if not isTune and args.fast_dev_run == 0:
        wandb_logger = WandbLogger(project='SSL')
    log_msg(args)

    if args.norm_p == -1.:
        args.norm_p = numpy.inf
    if args.distance_p == -1.:
        args.distance_p = numpy.inf

    if args.dataset == "cifar10": 
        dm = TinyCIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers) 
        args.num_samples = dm.num_samples
        normalization = cifar10_normalization()
    else:
        raise NotImplementedError("other datasets have not been implemented till now")
    
    print(dir(dm))
    

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=32,  # Standard dimension of CIFAR-10
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=32,#dm.dims[-1],
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    model = TinyExtractor()

    if args.fast_dev_run == 0:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="train_loss", dirpath=f"results/{wandb.run.name}/")
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