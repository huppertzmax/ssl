from training.losses.gamma_loss import gamma_loss
from training.losses.spectral_loss import spectral_loss
from training.losses.acos_kernel_loss import acos_kernel_loss
from training.losses.nt_xent_loss import nt_xent_loss
from training.losses.spectral_contrastive_loss import spectral_contrastive_loss
from training.losses.rq_min_loss import rq_min_loss
from training.losses.kernel_infonce import kernel_infonce_loss

def loss(out_1, out_2, temperature=0.1, gamma=2.0, gamma_lambd=1.0, distance_p=2.0, loss_type="nt_xent", acos_order=0, feat_dim=32, heat_kernel=None, penalty_constrained=False, graph_laplacian=None):
    if acos_order > 0:
        loss = acos_kernel_loss(out_1=out_1, out_2=out_2, temperature=temperature, acos_order=acos_order)
    else:
        if loss_type == "sum":
            loss = (
                gamma_loss(out_1=out_1, out_2=out_2, gamma=gamma, temperature=temperature, distance_p=distance_p) * gamma_lambd +
                gamma_loss(out_1=out_1, out_2=out_2, gamma=2.0, temperature=temperature, distance_p=distance_p) * (1. - gamma_lambd)
            )
        elif loss_type == "origin":
            loss = gamma_loss(out_1=out_1, out_2=out_2, gamma=gamma, temperature=temperature, distance_p=distance_p)
        elif loss_type == "product":
            loss = (
                gamma_loss(out_1=out_1[:, 0:feat_dim // 2], out_2=out_2[:, 0:feat_dim // 2], gamma=gamma,
                temperature=temperature, distance_p=distance_p) * gamma_lambd + 
                gamma_loss(out_1=out_1[:, feat_dim // 2: feat_dim], out_2=out_2[:, feat_dim // 2: feat_dim], gamma=2.0,
                temperature=temperature, distance_p=distance_p) * (1. - gamma_lambd)
            )
        elif loss_type == "spectral":
            loss = spectral_loss(out_1=out_1, out_2=out_2)
        elif loss_type == "nt_xent":
            loss = nt_xent_loss(out_1=out_1, out_2=out_2, temperature=temperature)
        elif loss_type == "spectral_contrastive":
            loss = spectral_contrastive_loss(out_1=out_1, out_2=out_2)
        elif loss_type == "rq_min":
            return rq_min_loss(out_1=out_1, out_2=out_2, heat_kernel=heat_kernel, penalty_constrained=penalty_constrained)
        elif loss_type == "kernel_infonce":
            return kernel_infonce_loss(out_1=out_1, out_2=out_2, graph_laplacian=graph_laplacian)
        else:
            raise NotImplementedError

    return loss