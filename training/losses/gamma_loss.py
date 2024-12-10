import torch


def gamma_loss(out_1, out_2, gamma, temperature, distance_p, eps=1e-6):
    out_1_dist = out_1
    out_2_dist = out_2

    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)
    cov = torch.pow(torch.cdist(out, out_dist, p=distance_p), gamma) * -1.
    
    sim = torch.exp(cov / temperature)
    neg = torch.clamp(sim.sum(dim=-1) - sim.diag(), min=eps)
    sim_adj = torch.pow(torch.norm(out_1 - out_2, dim=-1, p=distance_p), gamma) * -1.
    
    pos = torch.exp(sim_adj / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / (neg + eps)).mean()
    return loss