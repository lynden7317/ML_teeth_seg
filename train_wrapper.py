# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 20:21:14 2021

@author: lynden
"""

import os
import sys

model_folder = "./seg"
tids = [32] 
#[11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
#[31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]

for tid in tids:
    _stid = str(tid)
    output_folder = os.path.join(model_folder, "t"+_stid)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    cmd = "python train_seg.py --tid={} --outf={} > t{}_log.txt".format(_stid, output_folder, _stid)
    os.system(cmd)
