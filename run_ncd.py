import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from itertools import cycle
import numpy as np
import argparse
from arguments import set_deterministic, Namespace, csv, shutil, yaml
from augmentations import get_aug
from models import get_model
from optimizers import get_optimizer, LR_Scheduler
from datetime import date
from sklearn.cluster import KMeans
from ylib.ytool import cluster_acc
import open_world_cifar as datasets
from linear_probe import get_linear_acc

def main(log_writer, log_file, device, args):
    iter_count = 0

    dataroot = args.data_dir
    if args.dataset.name == 'cifar10':
        train_set_known = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs))
        train_set_novel = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs), unlabeled_idxs=train_set_known.unlabeled_idxs)
        train_set_known_eval = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
        train_set_novel_eval = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), unlabeled_idxs=train_set_known.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=False, train=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
        test_set_known = datasets.data.Subset(test_set, np.arange(len(test_set))[test_set.targets < args.labeled_num])
        test_set_novel = datasets.data.Subset(test_set, np.arange(len(test_set))[test_set.targets >= args.labeled_num])
        args.num_classes = 10
    elif args.dataset.name == 'cifar100':
        # known_class_division_1 = [
        #     "beaver", "dolphin", "otter", "seal", "whale", "aquarium_fish", "flatfish", "ray", "shark", "trout",
        #     "orchid", "poppy", "rose", "sunflower", "tulip", "bottle", "bowl", "can", "cup", "plate",
        #     "apple", "mushroom", "orange", "pear", "sweet_pepper", "clock", "keyboard", "lamp",
        #     "telephone", "television", "bed", "chair", "couch", "table", "wardrobe", "bee", "beetle", "butterfly",
        #     "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "bridge", "castle", "house", "road",
        #     "skyscraper"
        # ]
        #
        # known_class_division_2 = [
        #     "beaver", "dolphin", "otter", "aquarium_fish", "flatfish", "orchid", "poppy", "rose", "bottle",
        #     "bowl", "apple", "mushroom", "orange", "clock", "keyboard", "bed", "chair", "couch", "bee", "beetle",
        #     "bear", "leopard", "lion", "bridge", "castle", "cloud", "forest", "mountain", "camel", "cattle", "fox",
        #     "porcupine", "possum", "crab", "lobster", "baby", "boy", "girl", "crocodile", "dinosaur", "hamster",
        #     "mouse", "rabbit", "maple_tree", "oak_tree", "bicycle", "bus", "motorcycle", "lawn_mower", "rocket"
        # ]
        class_list = None

        train_set_known = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs), class_list=class_list)
        train_set_novel = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs), class_list=class_list, unlabeled_idxs=train_set_known.unlabeled_idxs)
        train_set_known_eval = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), class_list=class_list)
        train_set_novel_eval = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), class_list=class_list, unlabeled_idxs=train_set_known.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=False, train=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), class_list=class_list)
        test_set_known = datasets.data.Subset(test_set, np.arange(len(test_set))[test_set.targets < args.labeled_num])
        test_set_novel = datasets.data.Subset(test_set, np.arange(len(test_set))[test_set.targets >= args.labeled_num])
        args.num_classes = 100


    labeled_len = len(train_set_known)
    unlabeled_len = len(train_set_novel)
    labeled_batch_size = int(args.train.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    train_label_loader = torch.utils.data.DataLoader(train_set_known, batch_size=labeled_batch_size, shuffle=True, num_workers=4, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_set_novel, batch_size=args.train.batch_size - labeled_batch_size, shuffle=True, num_workers=4, drop_last=True)
    train_loader_known_eval = torch.utils.data.DataLoader(train_set_known_eval, batch_size=100, shuffle=True, num_workers=2)
    train_loader_novel_eval = torch.utils.data.DataLoader(train_set_novel_eval, batch_size=100, shuffle=True, num_workers=2)
    test_loader_known = torch.utils.data.DataLoader(test_set_known, batch_size=100, shuffle=False, num_workers=2)
    test_loader_novel = torch.utils.data.DataLoader(test_set_novel, batch_size=100, shuffle=False, num_workers=2)

    # define model
    model = get_model(args.model, args).to(device)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_label_loader),
        constant_predictor_lr=True
    )

    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for epoch in range(0, args.train.stop_at_epoch):

        #######################  Train #######################
        model.train()
        print("number of iters this epoch: {}".format(len(train_label_loader)))
        unlabel_loader_iter = cycle(train_unlabel_loader)
        for idx, ((x1, x2), target) in enumerate(train_label_loader):
            ((ux1, ux2), target_unlabeled) = next(unlabel_loader_iter)
            x1, x2, ux1, ux2, target, target_unlabeled = x1.to(device), x2.to(device), ux1.to(device), ux2.to(device), target.to(device), target_unlabeled.to(device)

            model.zero_grad()
            data_dict = model.forward_ncd(x1, x2, ux1, ux2, target)
            loss = data_dict['loss'].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            if (idx + 1) % args.print_freq == 0:
                if args.model.name == 'spectral':
                    loss1, loss2, loss3, loss4, loss5 = 0, data_dict["d_dict"]["loss2"].item(), 0, 0, data_dict["d_dict"]["loss5"].item()
                else:
                    loss1, loss2, loss3, loss4, loss5 = data_dict["d_dict"]["loss1"].item(), data_dict["d_dict"]["loss2"].item(), data_dict["d_dict"]["loss3"].item(), data_dict["d_dict"]["loss4"].item(), data_dict["d_dict"]["loss5"].item()

                print('Train: [{0}][{1}/{2}]\t Loss_all {3:.3f} \tc1:{4:.2e}\tc2:{5:.3f}\tc3:{6:.2e}\tc4:{7:.2e}\tc5:{8:.3f}'.format(
                    epoch, idx + 1, len(train_label_loader), loss.item(), loss1, loss2, loss3, loss4, loss5
                ))


        #######################  Evaluation #######################
        model.eval()

        def feat_extract(loader, layer='penul'):
            targets = np.array([])
            features = []
            for idx, (x, labels) in enumerate(loader):
                # feat = model.backbone.features(x.to(device, non_blocking=True))
                ret_dict = model.forward_eval(x.to(device, non_blocking=True), layer=layer)
                feat = ret_dict['features']
                targets = np.append(targets, labels.cpu().numpy())
                features.append(feat.data.cpu().numpy())
            return np.concatenate(features), targets.astype(int)


        if (epoch + 1) % args.deep_eval_freq == 0:

            normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

            features_train_k, ltrain_k = feat_extract(train_loader_known_eval, layer=args.layer)
            features_train_n, ltrain_n = feat_extract(train_loader_novel_eval, layer=args.layer)
            features_test_k, ltest_k = feat_extract(test_loader_known, layer=args.layer)
            features_test_n, ltest_n = feat_extract(test_loader_novel, layer=args.layer)

            ftrain_k = normalizer(features_train_k)
            ftrain_n = normalizer(features_train_n)
            ftest_k = normalizer(features_test_k)
            ftest_n = normalizer(features_test_n)

            #######################  Linear Probe #######################
            # lp_acc, _ = get_linear_acc(ftrain, ltrain, ftest, ltest_n, args.labeled_num, print_ret=False)
            lp_acc, (clf_known, _, _, lp_preds_k) = get_linear_acc(ftrain_k, ltrain_k, ftest_k, ltest_k, args.labeled_num, print_ret=False)

            #######################  K-Means #######################
            alg = KMeans(init="k-means++", n_clusters=args.num_classes - args.labeled_num, n_init=5, random_state=0)
            estimator = alg.fit(ftrain_n)
            kmeans_acc_train = cluster_acc(estimator.predict(ftrain_n), ltrain_n.astype(np.int64))
            kmeans_preds_test_n = estimator.predict(ftest_n)
            kmeans_acc_test = cluster_acc(kmeans_preds_test_n, ltest_n.astype(np.int64))

            kmeans_preds_all = np.concatenate([lp_preds_k.astype(np.int32), kmeans_preds_test_n + args.labeled_num])
            targets_all = np.concatenate([ltest_k, ltest_n])
            kmeans_overall_acc = cluster_acc(kmeans_preds_all, targets_all)

            write_dict = {
                'epoch': epoch,
                'lr': lr_scheduler.get_lr(),
                'kmeans_acc_train': kmeans_acc_train,
                'kmeans_acc_test': kmeans_acc_test,
                'kmeans_overall_acc': kmeans_overall_acc,
                'lp_acc': lp_acc
            }

            print(f"K-Means Train Acc: {kmeans_acc_train:.4f}\t K-Means Test ACC: {kmeans_acc_test:.4f}\t K-Means All Acc: {kmeans_overall_acc:.4f}\t linear Probe Acc: {lp_acc:.4f}")

            log_writer.writerow(write_dict)
            log_file.flush()

        #######################  Save Epoch #######################
        if (epoch + 1) % args.log_freq == 0:
            model_path = os.path.join(ckpt_dir, f"{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict()
            }, model_path)
            print(f"Model saved to {model_path}")

    # Save checkpoint
    model_path = os.path.join(ckpt_dir, f"latest_{epoch+1}.pth")
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, "checkpoints", f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', default='configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml', type=str)
    # parser.add_argument('-c', '--config-file', default='configs/spectral_resnet_mlp1000_norelu_cifar10_lr003_mu1.yaml', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--test_bs', type=int, default=80)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default='/home/sunyiyou/workspace/orca/datasets')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:10001')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--ckpt_dir', type=str, default='~/.cache/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--vis_freq', type=int, default=2000)
    parser.add_argument('--deep_eval_freq', type=int, default=50)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--labeled-num', default=80, type=int)
    parser.add_argument('--labeled-ratio', default=1, type=float)
    parser.add_argument('--gamma_l', default=0.0225, type=float)
    parser.add_argument('--gamma_u', default=2, type=float)
    parser.add_argument('--c3_rate', default=1, type=float)
    parser.add_argument('--c4_rate', default=2, type=float)
    parser.add_argument('--c5_rate', default=1, type=float)
    parser.add_argument('--proj_feat_dim', default=1000, type=int)
    parser.add_argument('--went', default=0.0, type=float)
    parser.add_argument('--momentum_proto', default=0.95, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--base_lr', default=0.03, type=float)
    parser.add_argument('--layer', default='penul', type=str)

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            if key not in vars(args):
                vars(args)[key] = value

    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    alpha = args.gamma_l
    beta = args.gamma_u
    scale = 1
    args.c1, args.c2 = 2 * alpha * scale, 2 * beta * scale
    args.c3, args.c4, args.c5 = alpha ** 2 * scale * args.c3_rate, \
                 alpha * beta * scale * args.c4_rate, \
                 beta ** 2 * scale * args.c5_rate

    args.train.base_lr = args.base_lr

    disc = f"labelnum-{args.labeled_num}-c1-{args.c1:.2f}-c2-{args.c2:.1f}-c3-{args.c3:.1e}-c4-{args.c4:.1e}-c5-{args.c5:.1e}-gamma_l-{args.gamma_l:.2f}-gamma_u-{args.gamma_u:.2f}-r345-{args.c3_rate}-{args.c4_rate}-{args.c5_rate}"+ \
           f"-lr{args.base_lr}-layer{args.layer}-seed{args.seed}"
    args.log_dir = os.path.join(args.log_dir, 'in-progress-'+'{}'.format(date.today())+args.name+'-{}'.format(disc))

    os.makedirs(args.log_dir, exist_ok=True)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    shutil.copy2(args.config_file, args.log_dir)
    set_deterministic(args.seed)

    vars(args)['aug_kwargs'] = {
        'name': args.model.name,
        'image_size': args.dataset.image_size
    }
    vars(args)['dataset_kwargs'] = {
        'dataset':args.dataset.name,
        'data_dir': args.data_dir,
        'download':args.download,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    log_file = open(os.path.join(args.log_dir, 'log.csv'), mode='w')
    fieldnames = ['epoch', 'lr', 'kmeans_acc_train', 'kmeans_acc_test', 'kmeans_overall_acc', 'lp_acc']
    log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    log_writer.writeheader()

    return args, log_file, log_writer


if __name__ == "__main__":
    args, log_file, log_writer = get_args()

    main(log_writer, log_file, device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')