import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import scipy.io as scio
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
# import h5py
import re
# import water
from test import test


from math import exp
from data_loader import ICDAR2015, Synth80k, ICDAR2013

###import file#######
# from augmentation import random_rot, crop_img_bboxes
# from gaussianmap import gaussion_transform, four_point_transform
# from generateheatmap import add_character, generate_target, add_affinity, generate_affinity, sort_box, real_affinity, generate_affinity_box
from mseloss import Maploss


from collections import OrderedDict
from eval.script import getresult


from PIL import Image
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from multiprocessing import Pool

from data_loader import CLASSES

# 3.2768e-5
random.seed(42)

# class SynAnnotationTransform(object):
#     def __init__(self):
#         pass
#     def __call__(self, gt):
#         image_name = gt['imnames'][0]
parser = argparse.ArgumentParser(description='CRAFT reimplementation')


parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=8, type=int,
                    help='batch size of training')
# parser.add_argument('--cdua', default=True, type=str2bool,
# help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')


args = parser.parse_args()


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (5 ** min(4, step)) * (0.98 ** max(0, step - 4))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    # gaussian = gaussion_transform()
    # box = scio.loadmat('/data/CRAFT-pytorch/syntext/SynthText/gt.mat')
    # bbox = box['wordBB'][0][0][0]
    # charbox = box['charBB'][0]
    # imgname = box['imnames'][0]7
    # imgtxt = box['txt'][0]

    #dataloader = syndata(imgname, charbox, imgtxt)
    # print("Hello?")
    dataloader = Synth80k('/datasets/SynthText/', target_size=768)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=14,
        shuffle=True,
        num_workers=14,
        drop_last=True,
        pin_memory=True)
    #batch_syn = iter(train_loader)
    # prefetcher = data_prefetcher(dataloader)
    # input, target1, target2 = prefetcher.next()
    # print(input.size())
    net = CRAFT()
    # net.load_state_dict(copyStateDict(torch.load('pretrain/synweights/synweights_front_cmtnd.pth')))
    # net.load_state_dict(copyStateDict(torch.load('/data/CRAFT-pytorch/1-7.pth')))
    # net.load_state_dict(copyStateDict(torch.load('/data/CRAFT-pytorch/craft_mlt_25k.pth')))
    # net.load_state_dict(copyStateDict(torch.load('vgg16_bn-6c64b313.pth')))
    #realdata = realdata(net)
    # realdata = ICDAR2015(net, '/data/CRAFT-pytorch/icdar2015', target_size = 768)
    # real_data_loader = torch.utils.data.DataLoader(
    #     realdata,
    #     batch_size=10,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True,
    #     pin_memory=True)
    net = net.cuda()
    #net = CRAFT_net

    # if args.cdua:
    net = torch.nn.DataParallel(net, device_ids=[0, ]).cuda()
    cudnn.benchmark = True
    # realdata = ICDAR2015(net, '/data/CRAFT-pytorch/icdar2015', target_size=768)
    # real_data_loader = torch.utils.data.DataLoader(
    #     realdata,
    #     batch_size=10,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True,
    #     pin_memory=True)

    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    criterion = Maploss()
    #criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    net.train()

    step_index = 0

    loss_time = 0
    loss_value = 0
    compare_loss = 1
    total_epochs = 5
    for epoch in range(total_epochs):
        loss_value = 0
        # if epoch % 50 == 0 and epoch != 0:
        #     step_index += 1
        #     adjust_learning_rate(optimizer, args.gamma, step_index)

        st = time.time()
        for i, (images, gh_label, mask, _) in enumerate(train_loader):
            # input()
            index = epoch * len(train_loader) + i
            if index % 100 == 0 and index != 0:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)
        
            # import imgproc
            # import cv2
            # import os
            
            # gh_permute = gh_label.permute((0,3,1,2))
            # for index in range(len(gh_permute)):
            #     for gh, field in zip(gh_permute[index], CLASSES):
            #         gh_img = gh.cpu().data.numpy()
            #         gh_img = imgproc.cvt2HeatmapImg(gh_img)
            #         img_path = os.path.join("prep", "gt_{}_{}.jpg".format(index, field))
            #         cv2.imwrite(img_path, gh_img)
            #     img = images[i].permute((1, 2, 0)).cpu().data.numpy()
            #     new_size = gh_img.shape[:2]
            #     new_size = new_size[::-1]
            #     img = cv2.resize(img, new_size)[::,::,::-1] * 255
            #     print(img.shape)
            #     print(img)
            #     img_path = os.path.join("prep", "gt_{}.jpg".format(index))
            #     cv2.imwrite(img_path, img)
            #     print('saved images')

            images = Variable(images.type(torch.FloatTensor)).cuda()
            
            # gh_label = gh_label.type(torch.FloatTensor)
            # gh_label = Variable(gh_label).cuda()
            
            gh_label = gh_label.type(torch.FloatTensor)
            gh_label = Variable(gh_label).cuda()
            
            mask = mask.type(torch.FloatTensor)
            mask = Variable(mask).cuda()

            out, _ = net(images)
            
            optimizer.zero_grad()

            out = out.cuda()

            # gh_permute = out.permute((0,3,1,2))
            # for index in range(len(gh_permute)):
            #     for gh, field in zip(gh_permute[index], CLASSES):
            #         gh_img = gh.cpu().data.numpy()
            #         gh_img = imgproc.cvt2HeatmapImg(gh_img)
            #         img_path = os.path.join("prep", "pred_{}_{}.jpg".format(index, field))
            #         cv2.imwrite(img_path, gh_img)
            #     print('saved pred')
            # input()
            
            # print(type(out))
            # print(type(gh_label))
            # print(out.cpu().data.numpy().shape)
            # print(gh_label.cpu().data.numpy().shape)
            loss = criterion(gh_label, out, mask)

            loss.backward()
            optimizer.step()
            loss_value += loss.item()

            PRINT_INTERVAL = 2
            if index % PRINT_INTERVAL == 0 and index > 0:
                et = time.time()
                print('epoch {}: ({} / {} / {}) batch || training time for {} batch: {:.2f} seconds || training loss {:.6f} ||'
                      .format(epoch, index, len(train_loader) * total_epochs, len(train_loader), PRINT_INTERVAL, et-st, loss_value/PRINT_INTERVAL))
                loss_time = 0
                loss_value = 0
                st = time.time()
            # if loss < compare_loss:
            #     print('save the lower loss iter, loss:',loss)
            #     compare_loss = loss
            #     torch.save(net.module.state_dict(),
            #                '/data/CRAFT-pytorch/real_weights/lower_loss.pth'

            if index % 1500 == 0 and index != 0:
                print('Saving state, index:', index)
                torch.save(net.module.state_dict(),
                           'pretrain/synweights_' + repr(index) + '.pth')
                # test('pretrain/synweights_' + repr(index) + '.pth')
                # test('/data/CRAFT-pytorch/craft_mlt_25k.pth')
                # getresult()
    print('Saving state, FINAL')
    torch.save(net.module.state_dict(),
                'pretrain/synweights.pth')
