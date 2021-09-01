# -*- coding: utf-8 -*-
"""
Created on Sun Aug 08 18:42:48 2021

@author: lynden
"""
from __future__ import print_function

import os
import random
import argparse
import numpy as np

import torch
import torch.optim as optim

from pointnet import dataset
from pointnet import model

import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=50, help='input batch size')  #32
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--nepoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default="./label", required=False, help="dataset path")

parser.add_argument('--npoints', type=int, default=1000, required=False, help="dataset path")
parser.add_argument('--tid', type=int, default=11, required=False, help="teeth id")
parser.add_argument('--feature_transform', default=True, action='store_true', help="use feature transform")


opt = parser.parse_args()
print(opt)

opt.feature_transform = True
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#print(dataset)
#sys.exit(1)
num_classes = 2
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = model.PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.0005, betas=(0.9, 0.999))  #0.001
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # step_size=20
classifier.cuda()

for epoch in range(opt.nepoch):
    scheduler.step()
    for root, dirs, files in os.walk(opt.dataset):
        if root == opt.dataset:
            continue
        
        #print(root)
        #sys.exit(1)
        db = dataset.ShapeNetDataset(
                    root=root,
                    tid=opt.tid,
                    npoints=opt.npoints)
        dataloader = torch.utils.data.DataLoader(
                    db,
                    batch_size=opt.batchSize,
                    shuffle=False, #True,
                    num_workers=int(opt.workers))
    
        num_batch = len(db) / opt.batchSize
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points = points.transpose(2, 1)
            batchSize = points.shape[0]
            #print(points.shape)
            #sys.exit(1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] #- 1
            #print(pred.size(), target.size(), target)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += model.feature_transform_regularizer(trans_feat) * 0.001
            
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            accuracy = correct.item()/float(batchSize*opt.npoints)
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), accuracy))
            #if i == 100:
            #    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))

    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))