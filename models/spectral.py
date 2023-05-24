import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50

def normalized_thresh(z, mu=1.0):
    if len(z.shape) == 1:
        mask = (torch.norm(z, p=2, dim=0) < np.sqrt(mu)).float()
        return mask * z + (1 - mask) * F.normalize(z, dim=0) * np.sqrt(mu)
    else:
        mask = (torch.norm(z, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        return mask * z + (1 - mask) * F.normalize(z, dim=1) * np.sqrt(mu)

def D(z1, z2, mu=1.0):
    mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    z1 = mask1 * z1 + (1-mask1) * F.normalize(z1, dim=1) * np.sqrt(mu)
    z2 = mask2 * z2 + (1-mask2) * F.normalize(z2, dim=1) * np.sqrt(mu)
    loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
    square_term = torch.matmul(z1, z2.T) ** 2
    loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                 z1.shape[0] / (z1.shape[0] - 1)
    return (loss_part1 + loss_part2) / mu, {"loss2": loss_part1 / mu, "loss5": loss_part2 / mu}


class projection_identity(nn.Module):
    def __init__(self):
        super().__init__()

    def set_layers(self, num_layers):
        pass

    def forward(self, x):
        return x


class Spectral(nn.Module):
    def __init__(self, backbone=resnet50(), mu=1.0, args=None):
        super().__init__()
        self.mu = mu
        self.args = args
        self.backbone = backbone
        self.projector = projection_identity()
        self.proto_num = self.args.dataset.numclasses - self.args.labeled_num
        self.register_buffer("label_stat", torch.zeros(self.proto_num, dtype=torch.int))

    # def forward(self, x1, x2, mu=1.0):
    #     f1, f2 = self.backbone(x1), self.backbone(x2)
    #     z1, z2 = self.projector(f1), self.projector(f2)
    #     L, d_dict = D(z1, z2, mu=self.mu)
    #     return {'loss': L, 'd_dict': d_dict}

    def forward_eval(self, x, proto_type=None, layer='penul'):
        penul_feat = self.backbone.features(x)
        proj_feat = normalized_thresh(self.projector(self.backbone.heads(penul_feat)))

        if layer == 'penul':
            feat = penul_feat
        else:
            feat = proj_feat

        return {
            "features": feat,
            "label_pseudo": torch.zeros(len(x)),
        }

    def forward_ncd(self, x1, x2, ux1, ux2, target, mu=1.0):
        x1 = torch.cat([x1, ux1], 0)
        x2 = torch.cat([x2, ux2], 0)
        f1, f2 = self.backbone(x1), self.backbone(x2)
        z1, z2 = self.projector(f1), self.projector(f2)
        L, d_dict = D(z1, z2, mu=self.mu)
        return {'loss': L, 'd_dict': d_dict}

    def forward(self, x1, x2, ux1, ux2, target, mu=1.0):
        return self.forward_ncd(x1, x2, ux1, ux2, target, mu)


    @torch.no_grad()
    def sync_prototype(self):
        pass

    @torch.no_grad()
    def reset_stat(self):
        self.label_stat = torch.zeros(self.proto_num, dtype=torch.int).to(self.label_stat.device)
