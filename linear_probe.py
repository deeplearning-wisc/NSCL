from __future__ import print_function

import os
import sys
import argparse
import time
import math

import easydict
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from ylib.ytool import ArrayDataset
cudnn.benchmark = True

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features, bn=False):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.bn = bn
        if bn:
            self.bn_layer = nn.BatchNorm1d(out_features)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        if self.bn:
            out = self.bn_layer(out)
        return out

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def train(train_loader, classifier, criterion, optimizer, epoch, print_freq=10):
    """one epoch training"""
    classifier.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (features, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        features = features.cuda(non_blocking=True).float()
        labels = labels.cuda(non_blocking=True).long()
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        output = classifier(features)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #
        # # print info
        # if (idx + 1) % print_freq == 0:
        #     print('Train: [{0}][{1}/{2}]\t'
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})\t'
        #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #            epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses, top1=top1))
        #     sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, classifier, criterion, print_freq):
    """validation"""
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    preds = np.array([])

    with torch.no_grad():
        end = time.time()
        for idx, (features, labels) in enumerate(val_loader):
            features = features.float().cuda()
            labels = labels.long().cuda()
            bsz = labels.shape[0]

            # forward
            # output = classifier(model.encoder(images))

            output = classifier(features.detach())
            loss = criterion(output, labels)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            preds = np.append(preds, pred.cpu().numpy())

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if (idx + 1) % print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #        idx, len(val_loader), batch_time=batch_time,
            #        loss=losses, top1=top1))

    # print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg, preds

def get_linear_acc(ftrain, ltrain, ftest, ltest, n_cls, epochs=50, args=None, classifier=None, print_ret=True, normed=False):

    cluster2label = np.unique(ltrain)
    label2cluster = {li: ci for ci, li in enumerate(cluster2label)}
    ctrain = [label2cluster[l] for l in ltrain]
    ctest = [label2cluster[l] for l in ltest]

    opt = easydict.EasyDict({
        "lr_decay_rate": 0.2,
        "cosine": False,
        "lr_decay_epochs": [30,60,90],
        "start_epoch": 0,
        "learning_rate": 5,
        "epochs": epochs,
        "print_freq": 200,
        "batch_size": 512,
        "momentum": 0.9,
        "weight_decay": 0,
    })
    if args is not None:
        for k, v in args.items():
            opt[k] = v

    best_acc = 0

    criterion = torch.nn.CrossEntropyLoss().cuda()
    if classifier is None:
        classifier = LinearClassifier(ftrain.shape[1], num_classes=n_cls).cuda()

    trainset = ArrayDataset(ftrain, labels=ctrain)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)

    valset = ArrayDataset(ftest, labels=ctest)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size, shuffle=False)

    optimizer = set_optimizer(opt, classifier)

    best_preds = None
    best_state = None
    # training routine
    for epoch in range(opt.start_epoch + 1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, classifier, criterion, optimizer, epoch, print_freq=opt.print_freq)
        time2 = time.time()
        # print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2 - time1, acc))
        # eval for one epoch
        loss, val_acc, preds = validate(val_loader, classifier, criterion, print_freq=opt.print_freq)
        if val_acc > best_acc:
            best_acc = val_acc
            best_preds = preds
            best_state = copy.deepcopy(classifier.state_dict())
        # print('epoch {}, best accuracy: {:.2f}'.format(epoch, best_acc))
    if print_ret:
        print(f'{acc:.2f}\t{best_acc:.2f}', end='\t')

    classifier.load_state_dict(best_state)
    return best_acc.item(), (classifier, cluster2label, label2cluster, best_preds)


def save_model(model, acc, save_file):
    print('==> Saving...')
    torch.save({
        'acc': acc,
        'state_dict': model.state_dict(),
    }, save_file)

