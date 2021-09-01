# -*- coding: utf-8 -*-
"""
Created on Mon Aug 09 21:59:46 2021

@author: lynden
"""

from __future__ import print_function

import os
import json
import re
import argparse
import numpy as np

import torch

from pointnet import dataset
from pointnet import model

import teeth_labeling

from teeth_stl import util
from teeth_stl import mesh
from teeth_stl import transformations
from teeth_stl import stl
from teeth_stl import conf
from teeth_stl import store

def loadConf(conf_path):
    try:
        with open(conf_path) as fin:
            loadJson = json.load(fin)
            return loadJson
    except IOError:
        print("[loadTeethConf] {} error".format(conf_path))
    
    return []
    

def teethcut_seg(tid, conf_path, tcut_path):
    fobj_teeth = open(tcut_path, "rb")
    stlTeeth = stl.load_stl(fobj_teeth)
    teethMesh = mesh.aaaMesh(vertices=stlTeeth['vertices'],
                             faces=stlTeeth['faces'],
                             face_normals=stlTeeth['face_normals'])
    
    centroid, rotateDegrees, centerDiff = teeth_labeling.meshAlign(conf_path, tid)
    tMatrix = transformations.translation_matrix([-centroid[0], -centroid[1], -centroid[2]])
    teethMesh.apply_transform(tMatrix)
    
    for r in rotateDegrees:
        rMatrix = transformations.rotation_matrix(r[0], r[1])
        teethMesh.apply_transform(rMatrix)
    
    if str(tid)[0] in ['1', '2']:
        teethMesh.apply_flip()
    
    # === find the maxz --> set to 0 === #
    maxz = teethMesh.bounds[1][2]
    zMatrix = transformations.translation_matrix([0.0, 0.0, -maxz])
    
    teethMesh.apply_transform(zMatrix)
    
    means = []
    for i, f in enumerate(teethMesh.triangles):
        _mean = np.mean(f, axis=0)
        #vertex = " ".join(["{:.6f}".format(p) for p in _mean])
        means.append(_mean)
    
    return teethMesh, np.array(means, dtype=np.float32), maxz

def teeth_reverse(teethMesh, tid, conf_path, maxz):
    zMatrix = transformations.translation_matrix([0.0, 0.0, maxz])
    teethMesh.apply_transform(zMatrix)
    
    if str(tid)[0] in ['1', '2']:
        teethMesh.apply_flip()
    
    centroid, rotateDegrees, centerDiff = teeth_labeling.meshAlign(conf_path, tid)
    
    #print(conf_path, rotateDegrees, type(rotateDegrees))
    rMatrix = transformations.rotation_matrix(-rotateDegrees[2][0], rotateDegrees[2][1])
    teethMesh.apply_transform(rMatrix)
    rMatrix = transformations.rotation_matrix(-rotateDegrees[1][0], rotateDegrees[1][1])
    teethMesh.apply_transform(rMatrix)
    rMatrix = transformations.rotation_matrix(-rotateDegrees[0][0], rotateDegrees[0][1])
    teethMesh.apply_transform(rMatrix)
    
    tMatrix = transformations.translation_matrix([centroid[0], centroid[1], centroid[2]])
    teethMesh.apply_transform(tMatrix)
    
    return teethMesh

def teeth_predict(tid, 
                  conf_path,
                  maxz,
                  teethMesh, 
                  meshMeans, 
                  output_folder, 
                  train_pth="./seg/t41/seg_model_49.pth"):
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    num_classes = 2
    feature_transform = True
    classifier = model.PointNetDenseCls(k=num_classes, feature_transform=feature_transform)
    
    if train_pth != '':
        classifier.load_state_dict(torch.load(train_pth))
    
    #print(meshMeans.shape)
    points = np.expand_dims(meshMeans, axis=0)
    points = points.transpose((0, 2, 1))
    points = torch.from_numpy(points)
    #print(points.shape, type(points))
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]
    
    pred_np = pred_choice.cpu().data.numpy()[0]
    #print(pred_np)
    
    fs, fnorm = [], []
    for i, f in enumerate(teethMesh.triangles):
        if pred_np[i] == 1:
            fs.append(f)
            fnorm.append(teethMesh.face_normals[i])
    
    #print(len(fs))
    if len(fs) > 0:
        extractMesh = mesh.aaaMesh(vertices=np.array(fs).reshape((-1, 3)),
                                   faces=np.arange(len(fs)*3).reshape((-1, 3)),
                                   face_normals=fnorm)
        
        extractMesh = teeth_reverse(extractMesh, tid, conf_path, maxz)
    
        output_path = os.path.join(output_folder, "t{}_pred.stl".format(tid))
        stl.export_mesh(extractMesh, output_path)
    else:
        print("{}: t{}_pred.stl no mesh is extracted".format(output_folder, tid))
    
    teethMesh = teeth_reverse(teethMesh, tid, conf_path, maxz)
    output_path = os.path.join(output_folder, "t{}_cut_align.stl".format(tid))
    stl.export_mesh(teethMesh, output_path)

def run_teeth_predict(test_folder, pred_folder, seg_tid):
    select_tcuts = []
    for root, dirs, files in os.walk(test_folder):
        for f in files:
            #par_folder = os.path.basename(root)
            test = re.match(r"^t([0-9]+).cut.stl", f)
            if test:
                tid = test.group(1)
                
                conf_path = os.path.join(root, "teeth.edgeRefined.conf")                
                tcut_path = os.path.join(root, f)
                if os.path.isfile(tcut_path):
                    if int(tid) == seg_tid:
                        select_tcuts.append([str(seg_tid), conf_path, tcut_path])
    
    train_path = "./seg/t{}/seg_model_8.pth".format(seg_tid)
    #train_path = "./seg/t{}/seg_model_29.pth".format(seg_tid)
    print(select_tcuts)
    for t in select_tcuts:
        _tid, _conf, _tcut = t
        if not os.path.isfile(_conf):
            print("{} not found".format(_conf))
            continue
        
        teethMesh, teethMeans, maxz = teethcut_seg(_tid, _conf, _tcut)
        #print(teethMeans.shape, type(teethMeans))
        output_folder = os.path.join(pred_folder, os.path.dirname(_tcut).split("\\")[-1])
        teeth_predict(_tid, 
                      _conf,
                      maxz,
                      teethMesh, 
                      teethMeans, 
                      output_folder,
                      train_pth=train_path)

def run_combine_teeth(test_folder, pred_folder):
    select_confs = []
    for root, dirs, files in os.walk(test_folder):
        conf_path = os.path.join(root, "teeth.edgeRefined.conf")
        if os.path.isfile(conf_path):
            select_confs.append(conf_path)
    
    for t in ["cut", "pred"]:    
        for conf_path in select_confs:
            teethJson = loadConf(conf_path)
            dirname = os.path.dirname(conf_path).split("\\")[-1]
            pred = os.path.join(pred_folder, dirname)
            print(conf_path, dirname)
            fs, fnorm = [], []
            for key, value in teethJson.items():
                if t == "cut":
                    stlPath = os.path.join(pred, "{}_cut_align.stl".format(key))
                else:
                    stlPath = os.path.join(pred, "{}_pred.stl".format(key))
                
                if os.path.isfile(stlPath):
                    fobj = open(stlPath, "rb")
                    stlmesh = stl.load_stl(fobj)
                    teethMesh = mesh.aaaMesh(vertices=stlmesh['vertices'],
                                             faces=stlmesh['faces'],
                                             face_normals=stlmesh['face_normals'])
                
                    #centroid = np.array([value["center"]["x"], value["center"]["y"], value["center"]["z"]])
                    #tMatrix = transformations.translation_matrix([centroid[0], centroid[1], centroid[2]])
                    #teethMesh.apply_transform(tMatrix)
                
                    for i, f in enumerate(teethMesh.triangles):
                        fs.append(f)
                        fnorm.append(teethMesh.face_normals[i])
        
            combineMesh = mesh.aaaMesh(vertices=np.array(fs).reshape((-1, 3)),
                                       faces=np.arange(len(fs)*3).reshape((-1, 3)),
                                       face_normals=fnorm)
            output_path = os.path.join(pred, "{}_total.stl".format(t))
            stl.export_mesh(combineMesh, output_path)

if __name__ == '__main__':
    #seg_tid = 41
    tid_list = [11, 12, 13, 14, 15, 16, 17,\
                21, 22, 23, 24, 25, 26, 27,\
                31, 32, 33, 34, 35, 36, 37,\
                41, 42, 43, 44, 45, 46, 47]
    tid_list = [32]
    cur_folder = os.getcwd()
    test_folder = os.path.join(cur_folder, "test")
    pred_folder = os.path.join(cur_folder, "predict")
    
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)
    
    if not os.path.isdir(pred_folder):
        os.makedirs(pred_folder)
    
    #for tid in tid_list:
    #    run_teeth_predict(test_folder, pred_folder, tid)
    
    run_combine_teeth(test_folder, pred_folder)
    
    """
    select_tcuts = []
    for root, dirs, files in os.walk(test_folder):
        for f in files:
            par_folder = os.path.basename(root)
            test = re.match(r"^t([0-9]+).cut.stl", f)
            if test:
                tid = test.group(1)
                
                conf_path = os.path.join(root, "teeth.edgeRefined.conf")                
                tcut_path = os.path.join(root, f)
                if os.path.isfile(tcut_path):
                    if int(tid) == seg_tid:
                        select_tcuts.append([str(seg_tid), conf_path, tcut_path])
                    
    
    print(select_tcuts)
    for t in select_tcuts:
        _tid, _conf, _tcut = t
        teethMesh, teethMeans = teethcut_seg(_tid, _conf, _tcut)
        print(teethMeans.shape, type(teethMeans))
        output_folder = os.path.join(pred_folder, os.path.dirname(_tcut).split("\\")[-1])
        teeth_predict(_tid, teethMesh, teethMeans, output_folder,
                      train_pth="./seg/t41/seg_model_49.pth")
    """