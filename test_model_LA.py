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
from vnet import VNet
from aleatoric import StochasticDeepMedic
import logging
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--al_weight', type=float, default=0.5, help='the weight of aleatoric uncertainty loss')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')

args = parser.parse_args()

al_weight = args.al_weight

res_dir = 'test_result/'

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
pretraining_epochs = 60
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



def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count




@torch.no_grad()
def test(net, val_loader, maxdice=0, save_result=False, test_save_path='./save'):
    metrics = test_calculate_metric_LA(net, val_loader.dataset, save_result=save_result, test_save_path=test_save_path)
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
    model_path = Path('/home/xiangjinyi/semi_supervised/alnet/LA_result/LA_0.8_al/con_5.0_consistency_1_VNet3')

    load_net_opt(net, optimizer, model_path / 'best.pth')
    #load_net_opt(ema_net, optimizer, pretrained_path / 'best.pth')
    # pretrain(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader, start_epoch=1)

    test(net, test_loader, save_result=True, test_save_path="/home/xiangjinyi/semi_supervised/alnet/LA_image_result_2/")

    #t_train(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader)

    logging.info(count_param(net))
