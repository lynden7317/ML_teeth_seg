# -*- coding: utf-8 -*-
"""
Created on Tue Aug 03 21:18:43 2021

@author: lynden
"""
import os
import torch
import torch.utils.data as data
import numpy as np

import math

# crowns: label=1, gums: label=0
class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 tid,
                 npoints=3000,
                 data_augmentation=False):
        self.npoints = npoints
        self.root = root
        self.tid = tid
        self.data_augmentation = data_augmentation
        
        self.point_set_crown = None
        self.point_set_gum = None
        
        self.dataset = {0:{"files":[]}, 1:{"files":[]}}
        for root, dirs, files in os.walk(self.root):
            # [gum, crown]
            record = [os.path.join(root, "t{}_gums.txt".format(self.tid)), os.path.join(root, "t{}_crowns.txt".format(self.tid))]
            for i, r in enumerate(record):
                if os.path.isfile(r):
                    self.dataset[i]["files"].append(r)
                    point_set = np.loadtxt(r).astype(np.float32)
                    
                    """
                    if self.point_set is not None:
                        self.point_set = np.concatenate((self.point_set, point_set), axis=0)
                        self.seg = np.concatenate((self.seg, np.array([i]*point_set.shape[0])), axis=0)
                    else:
                        self.point_set = np.array(point_set)
                        self.seg = np.array([i]*point_set.shape[0])
                    """
                    if i == 1:
                        if self.point_set_crown is not None:
                            self.point_set_crown = np.concatenate((self.point_set_crown, point_set), axis=0)
                        else:
                            self.point_set_crown = np.array(point_set)
                    
                    if i == 0:
                        if self.point_set_gum is not None:
                            self.point_set_gum = np.concatenate((self.point_set_gum, point_set), axis=0)
                        else:
                            self.point_set_gum = np.array(point_set)
                    
        #self.seg = np.array([self.seg.astype(np.int64)]).T        # reshape((len(self.seg), 1))
        
        self.num_seg_classes = len(self.dataset.keys())
        #print(self.dataset[1]["files"], self.dataset[0]["files"])
        #print(self.point_set, self.seg)
    
    def __getitem__(self, index):
        choice_crown = np.random.choice(self.point_set_crown.shape[0], int(self.npoints/2), replace=False)
        choice_gum = np.random.choice(self.point_set_gum.shape[0], int(self.npoints/2), replace=False)
        #resample
        point_set_crown = self.point_set_crown[choice_crown, :]
        point_set_gum = self.point_set_gum[choice_gum, :]
        point_set = np.concatenate((point_set_crown, point_set_gum), axis=0)
        seg = np.concatenate((np.array([1]*int(self.npoints/2)), np.array([0]*int(self.npoints/2))), axis=0)
        seg = np.array([seg]).T
        #print(point_set, seg)
        
        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi/36)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.1, size=point_set.shape) # random jitter 0.02
        
        tmp = np.hstack((point_set, seg))
        np.random.shuffle(tmp)
        tmp = np.hsplit(tmp, np.array([3, 6]))
        point_set = np.array(tmp[0], dtype=np.float32)
        seg = np.array(tmp[1], dtype=np.int64)
        #print(point_set, seg)
        
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        return point_set, seg
        
    """
    def __getitem__(self, index):
        choice = np.random.choice(len(self.seg), self.npoints, replace=True)
        #resample
        point_set = self.point_set[choice, :]
        
        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi/36)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.05, size=point_set.shape) # random jitter 0.02
        
        seg = self.seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        
        return point_set, seg
    """
    
    def __len__(self):
        return 1000
        #total = self.point_set_crown.shape[0] + self.point_set_gum.shape[0]
        #return math.ceil(total/self.npoints)-1 #math.ceil(self.seg.shape[0]/self.npoints)
    
    def __str__(self):
        zeros = self.point_set_gum.shape[0]
        ones = self.point_set_crown.shape[0]
        ratio = ones/zeros
        return "total: {}, #0: {}, #1: {}, #1/#0: {}".format(zeros+ones, zeros, ones, ratio)


if __name__ == '__main__':
    root = "./label"
    tid = 32
    db = ShapeNetDataset(root, tid, npoints=10, data_augmentation=False)
    print(len(db))
    
    ps, seg = db[0]
    print(ps, seg)
    print(ps.size(), ps.type(), seg.size(), seg.type())
    
    """
    if self.nstart == 0:
        print("shuffle point_set")
        tmp = np.hstack((self.point_set, self.seg))
        np.random.shuffle(tmp)
        tmp = np.hsplit(tmp, np.array([3, 6]))
        self.point_set = np.array(tmp[0])
        self.seg = np.array(tmp[1])
    """