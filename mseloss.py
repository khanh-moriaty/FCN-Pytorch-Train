import numpy as np
import torch
import torch.nn as nn


class Maploss(nn.Module):
    def __init__(self, use_gpu = True):

        super(Maploss,self).__init__()

    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1))*0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        internel = batch_size
        for i in range(batch_size):
            average_number = 0
            loss = torch.mean(pre_loss.view(-1)) * 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= 0.1)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < 0.1)])
                    average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < 0.1)], 3*positive_pixel)[0])
                    average_number += 3*positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                average_number += 500
                sum_loss += nega_loss
            #sum_loss += loss/average_number

        return sum_loss



    def forward(self, gh_label, gh_pred, mask):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        gh_label = gh_label.permute((3, 0, 1, 2))
        gh_pred = gh_pred.permute((3, 0, 1, 2))

        assert all([pred.size() == label.size() for (pred, label) in zip(gh_pred, gh_label)])
        
        # loss1 = loss_fn(p_gh, gh_label)
        # loss2 = loss_fn(p_gah, gah_label)
        # loss_g = torch.mul(loss1, mask)
        # loss_a = torch.mul(loss2, mask)
        
        loss = [loss_fn(pred, label) for (pred, label) in zip(gh_pred, gh_label)]
        loss = [torch.mul(ls, mask) for ls in loss]

        # char_loss = self.single_image_loss(loss_g, gh_label)
        # affi_loss = self.single_image_loss(loss_a, gah_label)
        
        single_loss = [self.single_image_loss(ls, label) for (ls, label) in zip(loss, gh_label)]
        res = sum([single_ls/ls.shape[0] for (single_ls, ls) in zip(single_loss, loss)])
        return res
        
        # return char_loss/loss_g.shape[0] + affi_loss/loss_a.shape[0]