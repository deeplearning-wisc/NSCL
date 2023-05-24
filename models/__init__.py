from .supspectral import SupSpectral
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2
from .backbones import resnet18_cifar_variant1_mlp1000_norelu, resnet18_cifar_variant1_mlp_norelu
from .backbones import resnet50_mlp8192_norelu_3layer, resnet50_mlp4096_norelu_3layer, resnet50_mlp2048_norelu_3layer,resnet50_mlp8192_norelu_2layer, resnet50_mlp4096_norelu_2layer, resnet50_mlp2048_norelu_2layer, resnet18_mlp1000_norelu


def get_backbone(backbone, castrate=True, proj_feat_dim=1000):
    backbone = eval(f"{backbone}(featdim={proj_feat_dim})")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg, args=None):
    if 'cifar' not in args.dataset.name:
        model_cfg.backbone = {
            (8192, 2): "resnet50_mlp8192_norelu_2layer",
            (4096, 2): "resnet50_mlp4096_norelu_2layer",
            (2048, 2): "resnet50_mlp2048_norelu_2layer",
            (8192, 3): "resnet50_mlp8192_norelu_3layer",
            (4096, 3): "resnet50_mlp4096_norelu_3layer",
            (2048, 3): "resnet50_mlp2048_norelu_3layer",
        }[(args.proj_feat_dim, args.proj_layers)]
    if model_cfg.name == 'spectral':
        if "mu" not in model_cfg.__dict__:
            model_cfg.mu = 1.0
        from .spectral import Spectral
        model = Spectral(get_backbone(model_cfg.backbone, args.proj_feat_dim), mu=model_cfg.mu, args=args)
    elif model_cfg.name == 'spectral_old':
        if "mu" not in model_cfg.__dict__:
            model_cfg.mu = 1.0
        from .spectral_old import Spectral
        model = Spectral(get_backbone(model_cfg.backbone), mu=model_cfg.mu)
    elif model_cfg.name == 'supspectral':
        if "mu" not in model_cfg.__dict__:
            model_cfg.mu = 1.0
        model = SupSpectral(get_backbone(model_cfg.backbone, args.proj_feat_dim), mu=model_cfg.mu, args=args)
    else:
        raise NotImplementedError
    return model






