import torch 


def spectral_loss(out_1, out_2, eps=1e-6):
    out_1_dist = out_1
    out_2_dist = out_2
    
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)
    cov = torch.pow(torch.mm(out, out_dist.t().contiguous()), 2)
    pos = torch.sum(torch.clamp(cov.sum(dim=-1) - cov.diag(), min=eps) * (1. / (out_1.shape[0] * (out_1.shape[0] - 1))))
    neg = torch.sum(out_1 * out_2) * (2. / (out_1.shape[0]))
    return pos - neg