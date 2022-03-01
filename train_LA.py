import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import utils1.loss
from dataset.make_dataset import make_data_3d
from dataset.LeftAtrium import LAHeart
from test_util import test_calculate_metric_LA
from utils1 import statistic, ramps
from utils1.loss import DiceLoss, SoftIoULoss
from utils1.losses import FocalLoss
from utils1.ResampleLoss import ResampleLossMCIntegral
from vnet import VNet
from aleatoric import StochasticDeepMedic
import logging
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--al_weight', type=float, default=0.8, help='the weight of aleatoric uncertainty loss')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')

args = parser.parse_args()

al_weight = args.al_weight

res_dir = 'LA_result/LA_{}_al/'.format(al_weight)

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

logging.basicConfig(filename=res_dir + "log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info('New Exp :')

# 2,1
# 因为加入了后面 aleatoric loss 的部分 gpu设置为多个点话会有问题
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# Parameters
num_class = 2
base_dim = 8

batch_size = 2
lr = 1e-3
beta1, beta2 = 0.5, 0.999


# log settings & test
pretraining_epochs = 40
self_training_epochs = 301
thres = 0.5
pretrain_save_step = 5
st_save_step = 10
pred_step = 10

r18 = False
split_name = 'LA_dataset'
data_root = '../LA_dataset'
cost_num = 3

alpha = 0.99
consistency = 1
consistency_rampup = 40


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        return self


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return ramps.sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


def create_model(ema=False):
    net = nn.DataParallel(VNet(n_branches=4))
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def get_model_and_dataloader():
    """Net & optimizer"""
    net = create_model()
    ema_net = create_model(ema=True).cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

    """Loading Dataset"""
    logging.info("loading dataset")

    trainset_lab = LAHeart(data_root, split_name, split='train_lab', require_mask=True)
    lab_loader = DataLoader(trainset_lab, batch_size=batch_size, shuffle=False, num_workers=0)

    trainset_unlab = LAHeart(data_root, split_name, split='train_unlab', no_crop=True)
    unlab_loader = DataLoader(trainset_unlab, batch_size=1, shuffle=False, num_workers=0)

    testset = LAHeart(data_root, split_name, split='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return net, ema_net, optimizer, lab_loader, unlab_loader, test_loader


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])
    logging.info('Loaded from {}'.format(path))


def transform_label(label):
    s = label.shape
    res = torch.zeros(s[0], 2, s[1], s[2], s[3]).cuda()

    mask = (label == 0).long().unsqueeze(1).cuda()
    res[:, 0, :, :, :][mask] = 1

    mask  = (label == 1).long().unsqueeze(1).cuda()
    res[:, 1, :, :, :][mask] = 1

    return res


def pretrain(net, ema_net, optimizer, start_epoch=1):

    trainset_lab = LAHeart(data_root, split_name, split='train_lab', require_mask=False)
    lab_loader = DataLoader(trainset_lab, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    testset = LAHeart(data_root, split_name, split='test', require_mask=False)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    save_path = Path(res_dir) / 'pretrain'
    save_path.mkdir(exist_ok=True)
    logging.info("Save path : {}".format(save_path))

    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S'))

    DICE = DiceLoss(nclass=2)
    #CE_con = nn.CrossEntropyLoss(weight=w_con.cuda())
    #CE_rad = nn.CrossEntropyLoss(weight=w_rad.cuda())
    Focal = FocalLoss()
    Iou = SoftIoULoss(nclass=2)

    maxdice1 = 0

    iter_num = 0

    for epoch in tqdm(range(start_epoch, pretraining_epochs + 1), ncols=70):
        logging.info('\n')
        """Testing"""
        if epoch % pretrain_save_step == 0:
            # maxdice, _ = test(net, unlab_loader, maxdice, max_flag)
            val_dice, maxdice1, max_flag = test(net, test_loader, maxdice1)

            writer.add_scalar('pretrain/test_dice', val_dice, epoch)

            save_net_opt(net, optimizer, save_path / ('%d.pth' % epoch), epoch)
            logging.info('Save model : {}'.format(epoch))
            if max_flag:
                save_net_opt(net, optimizer, save_path / 'best.pth', epoch)
                save_net_opt(ema_net, optimizer, save_path / 'best_ema.pth', epoch)

        train_loss, train_dice= \
            AverageMeter(), AverageMeter()
        net.train()
        for step, (img, lab) in enumerate(lab_loader):
            img, lab = img.cuda(), lab.cuda()
            out = net(img)


            ce_loss = F.cross_entropy(out[0], lab)
            dice_loss = DICE(out[1], lab)
            focal_loss = Focal(out[2], lab)

            # backup plan 直接label做unsqueeze(1
            iou_loss = Iou(out[3], lab)
            loss = (ce_loss + dice_loss + focal_loss + iou_loss) / 4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            masks = get_mask(out[0])
            train_dice.update(statistic.dice_ratio(masks, lab), 1)
            train_loss.update(loss.item(), 1)

            logging.info('epoch : %d, step : %d, train_loss: %.4f, train_dice: %.4f' % (epoch, step, train_loss.avg, train_dice.avg))


            writer.add_scalar('pretrain/loss_all', train_loss.avg, epoch * len(lab_loader) + step)
            writer.add_scalar('pretrain/train_dice', train_dice.avg, epoch * len(lab_loader) + step)
            update_ema_variables(net, ema_net, alpha, step)
        writer.flush()


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def get_mask(out):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).float()
    masks = masks[:, 1, :, :].contiguous()
    return masks


def train(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader):
    save_path = Path(res_dir) / 'LA_al_weight_{}'.format(al_weight)
    save_path.mkdir(exist_ok=True)
    logging.info("Save path : ", save_path)

    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S'))
    pretrained_path = Path(res_dir) / 'pretrain'

    # load already pretrained models
    pretrained_path = Path('/home/xiangjinyi/semi_supervised/alnet/LA_result/pancreas_VNet_0.5_al_cora/pretrain_con_5.0')
    load_net_opt(net, optimizer, pretrained_path / 'best.pth')
    load_net_opt(ema_net, optimizer, pretrained_path / 'best.pth')

    AL_module = nn.DataParallel(StochasticDeepMedic(num_classes=2))
    AL_module = AL_module.cuda()

    # load_net_opt(net, optimizer, save_path / 'best.pth')
    # load_net_opt(ema_net, optimizer, save_path / 'best.pth')

    consistency_criterion = utils1.loss.softmax_mse_loss

    DICE = DiceLoss(nclass=2)
    CE = nn.CrossEntropyLoss()

    Focal = FocalLoss()
    Iou = SoftIoULoss(nclass = 2)
    SSLoss = ResampleLossMCIntegral(20) # 原来论文用的20

    maxdice = 0
    maxdice1 = 0

    iter_num = 0
    new_loader, plab_dice = pred_unlabel(net, unlab_loader)
    writer.add_scalar('acc/plab_dice', plab_dice, 0)

    for epoch in tqdm(range(0, self_training_epochs)):
        logging.info('')
        writer.flush()
        if epoch % pred_step == 0:
            new_loader, plab_dice = pred_unlabel(net, unlab_loader)

        if epoch % st_save_step == 0:
            """Testing"""
            # val_dice, maxdice, _ = test(net, unlab_loader, maxdice)
            val_dice, maxdice1, max_flag = test(net, test_loader, maxdice1)
            writer.add_scalar('acc/plab_dice', plab_dice, epoch)

            """Save model"""
            if epoch > 0:
                save_net_opt(net, optimizer, str(save_path / ('{}.pth'.format(epoch))), epoch)
                logging.info('Save model : {}'.format(epoch))

            if max_flag:
                save_net_opt(net, optimizer, str(save_path / 'best.pth'), epoch)


        net.train()
        ema_net.train()
        for step, (data1, data2) in enumerate(zip(lab_loader, new_loader)):
            img1, lab1, lab_mask = data1
            img1, lab1, lab_mask = img1.cuda(), lab1.long().cuda(), lab_mask.long().cuda()
            img2, plab1, mask1, lab2 = data2
            img2, plab1, mask1 = img2.cuda(), plab1.long().cuda(), mask1.float().cuda()
            # plab2 = lab2.cuda()

            '''Supervised Loss'''
            out1 = net(img1)

            loss_ce1 = CE(out1[0], lab1)
            dice_loss1 = DICE(out1[1], lab1)
            focal_loss1 = Focal(out1[2], lab1)
            iou_loss1 = Iou(out1[3], lab1)

            logits, state = AL_module(net.module.al_input, lab_mask)
            state.update({'target': lab1})
            al_loss = SSLoss(logits, **state)

            # al_loss is computed in logit space. But the essence is cross entropy loss
            # So it's better to include in this averaging process
            supervised_loss = (loss_ce1 + focal_loss1 + iou_loss1 + dice_loss1 + al_loss * al_weight) / (4 + al_weight)


            # mask = torch.zeros_like(mask).cuda(mask.device).float()

            '''Certain Areas'''
            out2 = net(img2)
            loss_ce2 = (CE(out2[0], plab1) * mask1).sum() / (mask1.sum() + 1e-16)
            focal_loss2 = (Focal(out2[2], plab1) * mask1).sum() / (mask1.sum() + 1e-16)  #

            dice_loss2 = DICE(out2[1], plab1, mask1)
            iou_loss2 = Iou(out2[3], plab1, mask1)

            certain_loss = (loss_ce2 + dice_loss2 + focal_loss2 + iou_loss2) / 4


            '''Uncertain Areas---Mean Teacher'''
            mask1 = (1 - mask1).unsqueeze(1)
            with torch.no_grad():
                out_ema = ema_net(img2)
            consistency_weight = consistency * get_current_consistency_weight(epoch)
            consistency_dist1 = consistency_criterion(out2[0], out_ema[0])
            const_loss1 = consistency_weight * ((consistency_dist1 * mask1).sum() / (mask1.sum() + 1e-16))
            consistency_dist2 = consistency_criterion(out2[1], out_ema[1])
            const_loss2 = consistency_weight * ((consistency_dist2 * mask1).sum() / (mask1.sum() + 1e-16))
            consistency_dist3 = consistency_criterion(out2[2], out_ema[2])
            const_loss3 = consistency_weight * ((consistency_dist3 * mask1).sum() / (mask1.sum() + 1e-16))
            consistency_dist4 = consistency_criterion(out2[3], out_ema[3])
            const_loss4 = consistency_weight * ((consistency_dist4 * mask1).sum() / (mask1.sum() + 1e-16))
            uncertain_loss = (const_loss1 + const_loss2 + const_loss3 + const_loss4) / 4
            # logging.info(uncertain_loss)


            loss = supervised_loss + certain_loss + uncertain_loss   # uncertain_loss * 0.3 #+ certain_loss*0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                update_ema_variables(net, ema_net, alpha, iter_num + len(lab_loader) * pretraining_epochs)
                iter_num = iter_num + 1


        if epoch % st_save_step == 0:
            writer.add_scalar('val_dice', val_dice, epoch)


@torch.no_grad()
def pred_unlabel(net, pred_loader):
    logging.info('Starting predict unlab')
    unimg, unlab, unmask, labs = [], [], [], []
    plab_dice = 0
    for (step, data) in enumerate(pred_loader):
        img, lab = data
        img, lab = img.cuda(), lab.cuda()
        out = net(img)
        plab0 = get_mask(out[0]) # cross entropy prediction
        plab1 = get_mask(out[1]) # dice loss prediction
        plab2 = get_mask(out[2]) # focal loss prediction
        plab3 = get_mask(out[3]) # Iou loss prediction

        mask = ((plab0 == plab2) & (plab1 == plab3)).long()

        unimg.append(img)
        unlab.append(plab2) # suppose results derived from focal loss are the best
        unmask.append(mask)

        labs.append(lab)

        plab_dice += statistic.dice_ratio(plab2, lab)
    plab_dice /= len(pred_loader)
    logging.info('Pseudo label dice : {}'.format(plab_dice))
    new_loader1 = DataLoader(make_data_3d(unimg, unlab, unmask, labs), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    # new_loader2 = DataLoader(make_data(unimg2, unlab2), batch_size=batch_size, shuffle=True, num_workers=0)
    return new_loader1, plab_dice


@torch.no_grad()
def test(net, val_loader, maxdice=0):
    metrics = test_calculate_metric_LA(net, val_loader.dataset)
    val_dice = metrics[0]

    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False
    logging.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f' % (val_dice, maxdice))
    return val_dice, maxdice, max_flag


if __name__ == '__main__':
    # set_random_seed(1337)
    net, ema_net, optimizer, lab_loader, unlab_loader, test_loader = get_model_and_dataloader()
    # load model
    # net.load_state_dict(torch.load(res_dir + '/model/best.pth'))
    # pretrained_path = Path(res_dir) / 'pretrain_con_{}_consistency_{}'.format(w_con[1].item(), consistency)

    # load_net_opt(net, optimizer, pretrained_path / 'best.pth')
    # load_net_opt(ema_net, optimizer, pretrained_path / 'best.pth')
    #pretrain(net, ema_net, optimizer, start_epoch=1)

    train(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader)

    logging.info(count_param(net))
