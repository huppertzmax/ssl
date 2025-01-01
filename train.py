from argparse import ArgumentParser
from training.pretraining import pretraining
from training.tiny_pretraining import tiny_pretraining
from training.utils.utils import log_msg
import warnings

if __name__ == "__main__":

    parser = ArgumentParser()
    warnings.filterwarnings("ignore", message="The feature .* is currently marked under review.")

    #data 
    parser.add_argument("--dataset", type=str, default="cifar10", help="mnist, cifar10")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="path to download data")

    #model
    parser.add_argument("--arch", default="resnet18", type=str, help="convnet architecture")
    parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")

    # optimization 
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--learning_rate", default=0.25, type=float, help="base learning rate")
    parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")
    parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
    parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")

    # loss
    parser.add_argument("--norm_p", default=2., type=float, help="norm p, -1 for inf")
    parser.add_argument("--distance_p", default=2., type=float, help="p-norm distance, 2 for euclidean distance, -1 for inf")
    parser.add_argument("--acos_order", default=0, type=int, help="order of acos, 0 for not use acos kernel")
    parser.add_argument("--gamma", default=2., type=float, help="gamma")
    parser.add_argument("--gamma_lambd", default=1., type=float, help="gamma lambd")
    parser.add_argument("--projection_mu", default=1., type=float, help="projection mu")
    parser.add_argument("--loss_type", default="origin", type=str, help="nt_xent, origin, sum, product, spectral_contrastive or spectral")
    parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
    
    # transformation
    parser.add_argument("--gaussian_blur", default=False, action="store_true", help="add gaussian blur")
    parser.add_argument("--jitter_strength", type=float, default=0.5, help="jitter strength")

    # training
    parser.add_argument("--num_samples", default=1000, type=int, help="number of samples")
    parser.add_argument("--online_ft", default=False, action="store_true")
    parser.add_argument("--val_split", type=float, default=0.15, help="percentage of train data that is used for validation")
    parser.add_argument("--max_epochs", default=400, type=int, help="number of total epochs to run")
    parser.add_argument("--max_steps", default=200, type=int, help="max steps")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size per gpu")

    # system 
    parser.add_argument("--fast_dev_run", default=0, type=int)
    parser.add_argument("--accelerator", default="gpu", help="gpu or cpu")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
    parser.add_argument("--num_workers", default=4, type=int, help="num of workers per GPU")
    parser.add_argument("--fp32", default=True, action="store_true") 
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")

    parser.add_argument("--tiny", default=False, help="use tiny model")
    
    args = parser.parse_args()

    prefix = "Tiny pretraining" if args.tiny else "Pretraining"
    log_msg(f"{prefix} {args.arch} on {args.dataset} starting ...")
    if args.tiny:
        tiny_pretraining({}, args)
    pretraining({}, args)