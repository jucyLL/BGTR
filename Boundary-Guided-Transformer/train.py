import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
#from cityscapesscripts.helpers.labels import trainId2label
from torch import nn
from torch.backends import cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from dataset import creat_dataset, Cityscapes
from model import GatedSCNN
from utils import get_palette, compute_metrics, BoundaryBCELoss, DualTaskLoss

# for reproducibility
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False
maxmap = 0

# train or val or test for one epoch
def for_loop(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_time, total_num, preds, targets = 0.0, 0.0, 0, [], []
    data_bar = tqdm(data_loader, dynamic_ncols=True)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target, grad, boundary, name in data_bar: 
            torch.cuda.synchronize()
            start_time = time.time()
            seg, edge = net(data, grad)
            prediction = torch.argmax(seg.detach(), dim=1)
            torch.cuda.synchronize()
            end_time = time.time()
            semantic_loss = semantic_criterion(seg, target)
            edge_loss = edge_criterion(edge, target, boundary)
            task_loss = task_criterion(seg, edge, target)
            loss = semantic_loss + 20 * edge_loss + task_loss
            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_time += end_time - start_time
            total_loss += loss.item() * data.size(0)
            preds.append(prediction.cpu())
            targets.append(target.cpu())
        # compute metrics
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        pa, mpa, class_iou, category_iou = compute_metrics(preds, targets)
        #if mpa> maxmap:
        #    maxmap = mpa
        print('{} Epoch: [{}/{}] PA: {:.2f}% mPA: {:.2f}% Class_mIOU: {:.2f}% Category_mIOU: {:.2f}%'
              .format(data_loader.dataset.split.capitalize(), epoch, epochs,
                      pa * 100, mpa * 100, class_iou * 100, category_iou * 100))
        
    return total_loss / total_num, pa * 100, mpa * 100, class_iou * 100, category_iou * 100


if __name__ == '__main__':
    # train/val/test loop
    for epoch in range(1, epochs + 1):
        train_loss, train_PA, train_mPA, train_class_mIOU, train_category_mIOU = for_loop(model, train_loader,
                                                                                          optimizer)
        results['train_loss'].append(train_loss)
        results['train_PA'].append(train_PA)
        results['train_mPA'].append(train_mPA)
        results['train_class_mIOU'].append(train_class_mIOU)
        results['train_category_mIOU'].append(train_category_mIOU)
        val_loss, val_PA, val_mPA, val_class_mIOU, val_category_mIOU = for_loop(model, val_loader, None)
        results['val_loss'].append(val_loss)
        results['val_PA'].append(val_PA)
        results['val_mPA'].append(val_mPA)
        results['val_class_mIOU'].append(val_class_mIOU)
        results['val_category_mIOU'].append(val_category_mIOU)
        scheduler.step()
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_{}_{}_statistics.csv'.format(save_path, backbone_type, crop_h, crop_w),
                          index_label='epoch')
        if val_mPA > best_mpa:
            best_mpa = val_mPA
            print('best_mpa',best_mpa)
        if val_class_mIOU > best_mIOU:
            best_mIOU = val_class_mIOU
            # use best model to obtain the test results
            for_loop(model, test_loader, None)
            torch.save(model.state_dict(), '{}/{}_{}_{}_model.pth'.format(save_path, backbone_type, crop_h, crop_w))
