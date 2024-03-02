import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from networks.vnet import VNet
from utils import ramps, losses
from dataloaders.CMT_TS import CMT_TS, RandomCrop, RandomRotFlip, ToTensor, RandomNoise
from skimage.measure import label





parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='MVC-Net', help='model_name')
parser.add_argument('--max_iterations', type=int, default=12000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.002, help='maximum epoch number to train')
parser.add_argument('--lr_change', type=float, default=6000, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1 , help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=120.0, help='consistency_rampup')
parser.add_argument('--rampup_division',    type=float, default=120.0, help='rampup_division')
parser.add_argument('--consistensiy_weight',    type=float, default=100.0, help='consistensiy_weight')
args = parser.parse_args()


def getLargestCC(segmentation):
    labels = label(segmentation)
    # assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() == 0:
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

rampup_division = args.rampup_division
consistensiy_weight = args.consistensiy_weight
num_classes = 5
patch_size = (112, 112, 112)
lr_changes = args.lr_change

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def create_model(ema=False):
    # Network definition
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


if __name__ == "__main__":
    model1 = create_model()
    model2 = create_model()
    model3 = create_model()


    patch_size = [112, 112, 112]
    train_ds = CMT_TS(transform=transforms.Compose([
        RandomCrop(patch_size),
        RandomRotFlip(),
        RandomNoise(),
        ToTensor()
    ]))
    trainloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model1.train()
    model2.train()
    model3.train()
    optimizer1 = optim.Adam(model1.parameters(), lr=base_lr)
    optimizer2 = optim.Adam(model2.parameters(), lr=base_lr)
    optimizer3 = optim.Adam(model3.parameters(), lr=base_lr)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    model1.train()
    model2.train()
    model3.train()

    best_dice1 = 0
    best_dice2 = 0
    best_dice3 = 0

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, batch in enumerate(trainloader):
            print('i_batch', i_batch)
            labeled_image, slabel, unlabeled_volume_batch = (
                batch['image'].cuda(), batch['label'].cuda(), batch['unlabel_image'].cuda())

            volume_batch = torch.cat((labeled_image, unlabeled_volume_batch), dim=0)
            label_batch = torch.cat((slabel, slabel), dim=0)

            volume_batch1 = volume_batch
            volume_batch2 = volume_batch + torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            volume_batch3 = volume_batch + torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)

            outputs1 = model1(volume_batch1)
            outputs2 = model2(volume_batch2.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            outputs3 = model3(volume_batch3.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)

            loss_seg1 = F.cross_entropy(outputs1[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft1 = F.softmax(outputs1, dim=1)
            loss_seg_dice1_1 = losses.dice_loss(outputs_soft1[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_dice2_1 = losses.dice_loss(outputs_soft1[:labeled_bs, 2, :, :, :], label_batch[:labeled_bs] == 2)
            loss_seg_dice3_1 = losses.dice_loss(outputs_soft1[:labeled_bs, 3, :, :, :], label_batch[:labeled_bs] == 3)
            loss_seg_dice4_1 = losses.dice_loss(outputs_soft1[:labeled_bs, 4, :, :, :], label_batch[:labeled_bs] == 4)
            loss_seg_dice1 = (loss_seg_dice1_1 + loss_seg_dice2_1 + loss_seg_dice3_1 + 2 * loss_seg_dice4_1) / 5
            supervised_loss1 = 0.2 * loss_seg1 + 0.8 * loss_seg_dice1

            loss_seg2 = F.cross_entropy(outputs2[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft2 = F.softmax(outputs2, dim=1)
            loss_seg_dice1_2 = losses.dice_loss(outputs_soft2[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_dice2_2 = losses.dice_loss(outputs_soft2[:labeled_bs, 2, :, :, :], label_batch[:labeled_bs] == 2)
            loss_seg_dice3_2 = losses.dice_loss(outputs_soft2[:labeled_bs, 3, :, :, :], label_batch[:labeled_bs] == 3)
            loss_seg_dice4_2 = losses.dice_loss(outputs_soft2[:labeled_bs, 4, :, :, :], label_batch[:labeled_bs] == 4)
            loss_seg_dice2 = (loss_seg_dice1_2 + loss_seg_dice2_2 + loss_seg_dice3_2 + 2 * loss_seg_dice4_2) / 5
            supervised_loss2 = 0.2 * loss_seg2 + 0.8 * loss_seg_dice2

            loss_seg3 = F.cross_entropy(outputs3[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft3 = F.softmax(outputs3, dim=1)
            loss_seg_dice1_3 = losses.dice_loss(outputs_soft3[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_dice2_3 = losses.dice_loss(outputs_soft3[:labeled_bs, 2, :, :, :], label_batch[:labeled_bs] == 2)
            loss_seg_dice3_3 = losses.dice_loss(outputs_soft3[:labeled_bs, 3, :, :, :], label_batch[:labeled_bs] == 3)
            loss_seg_dice4_3 = losses.dice_loss(outputs_soft3[:labeled_bs, 4, :, :, :], label_batch[:labeled_bs] == 4)
            loss_seg_dice3 = (loss_seg_dice1_3 + loss_seg_dice2_3 + loss_seg_dice3_3 + 2 * loss_seg_dice4_3) / 5
            supervised_loss3 = 0.2 * loss_seg3 + 0.8 * loss_seg_dice3


            mask1 = outputs_soft1[labeled_bs:]
            mask2 = outputs_soft2[labeled_bs:]
            mask3 = outputs_soft3[labeled_bs:]

            mask1 = torch.argmax(mask1, dim=1)
            mask2 = torch.argmax(mask2, dim=1)
            mask3 = torch.argmax(mask3, dim=1)

            mask1 = torch.where(mask1 == 0, torch.tensor(0), torch.tensor(1))
            mask2 = torch.where(mask2 == 0, torch.tensor(0), torch.tensor(1))
            mask3 = torch.where(mask3 == 0, torch.tensor(0), torch.tensor(1))

            mask = torch.logical_and(torch.logical_and(mask1.to(torch.bool), mask2.to(torch.bool)),mask3.to(torch.bool)).int()
            zeros_count = torch.sum(mask == 0)
            ones_count = torch.sum(mask == 1)
            mask = mask.cpu().data.numpy()

            for i in range(2):
                mask[i,:,:,:] = getLargestCC(mask[i,:,:,:])

            mask =  torch.from_numpy(mask).cuda()

            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, 5, 1, 1, 1)  # 在第一个维度上重复5次



            consistency_loss1 = consistency_criterion(outputs1[labeled_bs:], outputs2[labeled_bs:]) * mask
            consistency_loss2 = consistency_criterion(outputs1[labeled_bs:], outputs3[labeled_bs:]) * mask
            consistency_loss3 = consistency_criterion(outputs2[labeled_bs:], outputs3[labeled_bs:]) * mask

            unsupervised_loss1 = (torch.mean(consistency_loss1) + torch.mean(consistency_loss2)) * consistensiy_weight
            unsupervised_loss2 = (torch.mean(consistency_loss2) + torch.mean(consistency_loss3)) * consistensiy_weight
            unsupervised_loss3 = (torch.mean(consistency_loss3) + torch.mean(consistency_loss1)) * consistensiy_weight

            print(unsupervised_loss1)
            print(unsupervised_loss2)
            print(unsupervised_loss3)


            consistency_weight = get_current_consistency_weight(iter_num // rampup_division)

            loss1 = supervised_loss1 + consistency_weight * unsupervised_loss1
            loss2 = supervised_loss2 + consistency_weight * unsupervised_loss2
            loss3 = supervised_loss3 + consistency_weight * unsupervised_loss3



            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            loss1.backward(retain_graph=True)
            optimizer1.step()

            loss2.backward(retain_graph=True)
            optimizer2.step()

            loss3.backward()
            optimizer3.step()


            iter_num = iter_num + 1

            logging.info('iteration %d : loss : %f loss_supervised : %f' %
                         (iter_num, loss1.item(), supervised_loss1.item()))

            writer.add_scalar('loss/loss', loss1, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg1, iter_num)
            writer.add_scalar('loss/loss_seg_dice1', loss_seg_dice1, iter_num)
            writer.add_scalar('loss/loss_seg_dice2', loss_seg_dice2, iter_num)
            writer.add_scalar('loss/loss_seg_dice3', loss_seg_dice3, iter_num)

            writer.add_scalar('train/unsupervised_loss1', consistency_weight*unsupervised_loss1, iter_num)
            writer.add_scalar('train/unsupervised_loss2', consistency_weight*unsupervised_loss2, iter_num)
            writer.add_scalar('train/unsupervised_loss3', consistency_weight*unsupervised_loss3, iter_num)


            ## change lr
            if iter_num % lr_changes == 0:
                lr_ = base_lr * 0.1 ** (iter_num // lr_changes)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer3.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path1 = os.path.join(snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path1)
                logging.info("save model to {}".format(save_mode_path1))

                save_mode_path2 = os.path.join(snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path2)
                logging.info("save model to {}".format(save_mode_path2))

                save_mode_path3 = os.path.join(snapshot_path, 'model3_iter_' + str(iter_num) + '.pth')
                torch.save(model3.state_dict(), save_mode_path3)
                logging.info("save model to {}".format(save_mode_path3))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model1.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
