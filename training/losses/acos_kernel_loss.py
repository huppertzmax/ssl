import numpy 
import torch

def acos_kernel_distance(acos_order, angle):
    if acos_order == 1:
        dis = numpy.pi - angle
    elif acos_order == 2:
        dis = torch.sin(angle) + (numpy.pi - angle) * torch.cos(angle)
    elif acos_order == 3:
        dis = torch.sin(angle) * torch.cos(angle) * 3. + (numpy.pi - angle) * (
                    1 + torch.cos(angle) * torch.cos(angle) * 2.)
    else:
        raise NotImplementedError
    return dis

def acos_kernel_loss(out_1, out_2, temperature, acos_order, eps=1e-6):
    out_1_dist = out_1
    out_2_dist = out_2

    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)


    sim = acos_kernel_distance(acos_order, torch.acos(temperature * torch.mm(out, out_dist.t().contiguous()) + 1 - temperature + eps))
    neg = torch.clamp(sim.sum(dim=-1) - sim.diag(), min=eps)
    pos = acos_kernel_distance(acos_order, torch.acos(temperature * torch.sum(out_1 * out_2, dim=-1) + 1 - temperature + eps))
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / (neg + eps)).mean()
    
    return loss