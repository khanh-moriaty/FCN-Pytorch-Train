"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import shutil

from craft import CRAFT
from data_loader import CLASSES

from collections import OrderedDict
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

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
# parser.add_argument('--trained_model', default='pretrain/synweights/synweights_front_cmtnd.pth', type=str, help='pretrained model')
parser.add_argument('--trained_model', default='pretrain/synweights.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.65, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.65, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--canvas_size', default=768, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=True, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='data/', type=str, help='folder path to input images')

args = parser.parse_args()


""" For test images in a folder """
# image_list, _, _ = file_utils.get_files('/storage/upload_complete/')
# image_list, _, _ = file_utils.get_files('/storage/prep/')
image_list, _, _ = file_utils.get_files('/dataset/crawl/front_cmtnd_resized/test/')
print(len(image_list))

result_folder = '/storage/result/'
os.makedirs(result_folder, exist_ok=True)
shutil.rmtree(result_folder)
os.makedirs(result_folder, exist_ok=True)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, _ = net(x)

    # # make score and link map
    # score_text = y[0,:,:,0].cpu().data.numpy()
    # score_link = y[0,:,:,1].cpu().data.numpy()
    
    gh_pred = y[0, :, :, :].permute((2, 0, 1)).cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    boxes, polys = None, None

    # # Post-processing
    # boxes, polys = craft_utils.getDetBoxes(score_text, text_threshold, low_text, poly)
    postproc = [craft_utils.getDetBoxes(score_text, text_threshold, low_text, poly) for score_text in gh_pred]
    boxes_pred, polys_pred = zip(*postproc)

    # # coordinate adjustment
    # boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    # polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    
    for boxes, polys in zip(boxes_pred, polys_pred):
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # # render results (optional)
    # render_img = score_text.copy()
    # render_img = np.hstack((render_img, score_link))
    # ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return gh_pred, boxes_pred, polys_pred, size_heatmap

    return boxes, polys, ret_score_text

import gc


def test(modelpara):
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint {}'.format(modelpara))
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(modelpara)))
    else:
        net.load_state_dict(copyStateDict(torch.load(modelpara, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\n')
        image = imgproc.loadImage(image_path)
        res = image.copy()

        # bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)
        gh_pred, bboxes_pred, polys_pred, size_heatmap = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)
        
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        result_dir = os.path.join(result_folder, filename)
        os.makedirs(result_dir, exist_ok=True)
        for gh_img, field in zip(gh_pred, CLASSES):
            img = imgproc.cvt2HeatmapImg(gh_img)
            img_path = os.path.join(result_dir, 'res_{}_{}.jpg'.format(filename, field))
            cv2.imwrite(img_path, img)
        h, w = image.shape[:2]
        img = cv2.resize(image, size_heatmap)[::,::,::-1]
        img_path = os.path.join(result_dir, 'res_{}.jpg'.format(filename, field))
        cv2.imwrite(img_path, img)
        
        # # save score text
        # filename, file_ext = os.path.splitext(os.path.basename(image_path))
        # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        # cv2.imwrite(mask_file, score_text)
        
        res = cv2.resize(res, size_heatmap)
        for polys, field in zip(polys_pred, CLASSES):
            TEXT_WIDTH = 10*len(field)+10
            TEXT_HEIGHT = 15
            polys = np.int32([poly.reshape((-1, 1, 2)) for poly in polys])
            res = cv2.polylines(res, polys, True, (0, 0, 255), 2)
            for poly in polys:
                poly[1,0] = [poly[0,0,0]-10, poly[0,0,1]]
                poly[2,0] = [poly[0,0,0]-10, poly[0,0,1]+TEXT_HEIGHT]
                poly[3,0] = [poly[0,0,0]-TEXT_WIDTH, poly[0,0,1]+TEXT_HEIGHT]
                poly[0,0] = [poly[0,0,0]-TEXT_WIDTH, poly[0,0,1]]
            res = cv2.fillPoly(res, polys, (224,224,224))
            # print(poly)
            for poly in polys:
                res = cv2.putText(res, field, tuple(poly[3,0]+[+5,-5]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), thickness=1)
        res_file = os.path.join(result_dir, 'res_{}_bbox.jpg'.format(filename, field))
        cv2.imwrite(res_file, res[::,::,::-1])
        # break

        # file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))

if __name__ == "__main__":
    test(args.trained_model)
