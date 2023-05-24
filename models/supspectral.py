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

normalizer = lambda x: x / torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-10

class projection_identity(nn.Module):
    def __init__(self):
        super().__init__()

    def set_layers(self, num_layers):
        pass

    def forward(self, x):
        return x


class SupSpectral(nn.Module):
    def __init__(self, backbone=resnet50(), mu=1.0, args=None):
        super().__init__()
        self.mu = mu
        self.args = args
        self.backbone = backbone
        self.projector = projection_identity()

    def D(self, z1, z2, uz1, uz2, target, mu=1.0):
        device = z1.device
        bsz_l, bsz_u = len(z1), len(uz1)

        mat_ll = torch.matmul(z1, z2.T)
        mat_uu = torch.matmul(uz1, uz2.T)

        mat_lu_s2 = torch.matmul(z1, uz2.T) ** 2
        mat_ul_s2 = torch.matmul(uz1, z2.T) ** 2
        mat_ll_s2 = mat_ll ** 2 * (1 - torch.diag(torch.ones(bsz_l)).to(device))
        mat_uu_s2 = mat_uu ** 2 * (1 - torch.diag(torch.ones(bsz_u)).to(device))

        c1, c2 = self.args.c1, self.args.c2
        c3, c4, c5 = self.args.c3, self.args.c4, self.args.c5

        target_ = target.contiguous().view(-1, 1)
        pos_labeled_mask = torch.eq(target_, target_.T).to(device)
        cls_sample_count = pos_labeled_mask.sum(1)

        loss1 = - c1 * torch.sum((mat_ll * pos_labeled_mask) / cls_sample_count ** 2)

        pos_unlabeled_mask = torch.diag(torch.ones(bsz_u)).to(device)
        loss2 = - c2 * torch.sum(mat_uu * pos_unlabeled_mask) / bsz_u

        loss3 = c3 * torch.sum(mat_ll_s2 / (cls_sample_count[:, None] * cls_sample_count[None, :]))

        loss4 = c4 * torch.sum(mat_lu_s2 / (cls_sample_count[:, None] * bsz_u)) + \
                c4 * torch.sum(mat_ul_s2 / (cls_sample_count[None, :] * bsz_u))

        loss5 = c5 * torch.sum(mat_uu_s2) / (bsz_u * (bsz_u - 1))

        return (loss1 + loss2 + loss3 + loss4 + loss5) / mu, {"loss1": loss1 / mu, "loss2": loss2 / mu,
                                                              "loss3": loss3 / mu, "loss4": loss4 / mu,
                                                              "loss5": loss5 / mu}

    def forward_eval(self, x, layer='penul'):
        penul_feat = self.backbone.features(x)
        proj_feat = normalized_thresh(self.projector(self.backbone.heads(penul_feat)))
        if layer == 'penul':
            feat = penul_feat
        else:
            feat = proj_feat
        return {
            "features": feat,
        }

    def forward(self, x1, x2, ux1, ux2, target, mu=1.0):
        return self.forward_ncd(x1, x2, ux1, ux2, target, mu)

    def forward_ncd(self, x1, x2, ux1, ux2, target, mu=1.0):
        x = torch.cat([x1, x2, ux1, ux2], 0)
        penul_feat = self.backbone.features(x)
        proj_feat = normalized_thresh(self.projector(self.backbone.heads(penul_feat)))
        z = proj_feat
        # z = self.projector(self.backbone(x))
        # z = normalized_thresh(z, mu)
        z1 = z[0:len(x1), :]
        z2 = z[len(x1):len(x1)+len(x2), :]
        uz1 = z[len(x1)+len(x2):len(x1)+len(x2)+len(ux1), :]
        uz2 = z[len(x1)+len(x2)+len(ux1):, :]

        spec_loss, d_dict = self.D(z1, z2, uz1, uz2, target, mu=self.mu)

        loss = spec_loss
        return {'loss': loss, 'd_dict': d_dict}
