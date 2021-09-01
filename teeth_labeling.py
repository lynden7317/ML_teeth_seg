# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:47:26 2021

@author: lynden
"""
import os
import re
import math
import json
import numpy as np

from numpy import linalg as LA

from teeth_stl import util
from teeth_stl import mesh
from teeth_stl import transformations
from teeth_stl import stl
from teeth_stl import conf
from teeth_stl import store


def meshAlign(teethConf, tid):
    """
    Parameter
    ----------
    tid: 1x, 2x, 3x, 4x
        
    Return
    ----------
    centroid: list [x, y, z]
    rotateDegrees: list [(X-angle, [1,0,0]), (Y-angle, [0,1,0]), (Z-angle, [0,0,1])]
    """
    #teethConf = 'teeth.edgeRefined.conf' #'teeth.conf'
    try:
        with open(teethConf) as infile:
            teethJSON = json.load(infile)
    except IOError:
        print('Error: <%s> not found!' %teethConf)
        
    #print('{} read Conf, path: {}'.format(datetime.now(), (self.openDir + '/' + teethConf)))
    print('read Conf, path: {}'.format(teethConf))
        
    tt = 't'+str(tid)
        
    #centroid = [teethJSON[tt]['center']['x'], teethJSON[tt]['center']['y'], \
    #            teethJSON[tt]['center']['z']+teethJSON[tt]['axesCenterDiff']['z']]
    if tt in teethJSON.keys():
        centroid = [teethJSON[tt]['center']['x']+teethJSON[tt]['axesCenterDiff']['x'], 
                    teethJSON[tt]['center']['y']+teethJSON[tt]['axesCenterDiff']['y'],
                    teethJSON[tt]['center']['z']+teethJSON[tt]['axesCenterDiff']['z']]
        
        centerDiff = [teethJSON[tt]['axesCenterDiff']['x'], 
                      teethJSON[tt]['axesCenterDiff']['y'],
                      teethJSON[tt]['axesCenterDiff']['z']]
        
        R = conf.selectRotationMatrix(teethJSON[tt])
    else:
        print('Warning: t{} not in the configuration, set to default values'.format(tid))
        centroid = [0.0, 0.0, 0.0]
        centerDiff = [0.0, 0.0, 0.0]
        R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        
        
    angles = transformations.rotationMatrixToEulerAngles(R)
    rotateDegrees = [(angles[0], [1,0,0]), (angles[1], [0,1,0]), (angles[2], [0,0,1])]
        
    #print '*** centroid: ', centroid
    #print '*** rotateDegrees: ', rotateDegrees
    
    return centroid, rotateDegrees, centerDiff

def teethcut_align(tid, conf_path, tcut_path, output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
        
    fobj_teeth = open(tcut_path, "rb")
    stlTeeth = stl.load_stl(fobj_teeth)
    teethMesh = mesh.aaaMesh(vertices=stlTeeth['vertices'],
                             faces=stlTeeth['faces'],
                             face_normals=stlTeeth['face_normals'])
    
    centroid, rotateDegrees, centerDiff = meshAlign(conf_path, tid)
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
        vertex = " ".join(["{:.6f}".format(p) for p in _mean])
        means.append(vertex)
    
    with open(os.path.join(output_folder, "t{}_means.txt".format(tid)), "w") as fid:
        fid.write("\n".join(means))



def teethmesh_labeling(tid, conf_path, tcut_path, texport_path, label_folder):
    if not os.path.isdir(label_folder):
        os.makedirs(label_folder)
    
    fobj_teeth = open(tcut_path, "rb")
    stlTeeth = stl.load_stl(fobj_teeth)
    teethMesh = mesh.aaaMesh(vertices=stlTeeth['vertices'],
                             faces=stlTeeth['faces'],
                             face_normals=stlTeeth['face_normals'])
    
    fobj_teeth = open(texport_path, "rb")
    labelTeeth = stl.load_stl(fobj_teeth)
    labeledMesh = mesh.aaaMesh(vertices=labelTeeth['vertices'],
                               faces=labelTeeth['faces'],
                               face_normals=labelTeeth['face_normals'])
    
    centroid, rotateDegrees, centerDiff = meshAlign(conf_path, tid)
    tMatrix = transformations.translation_matrix([-centroid[0], -centroid[1], -centroid[2]])
    teethMesh.apply_transform(tMatrix)
    
    for r in rotateDegrees:
        rMatrix = transformations.rotation_matrix(r[0], r[1])
        teethMesh.apply_transform(rMatrix)
        labeledMesh.apply_transform(rMatrix)
    
    if str(tid)[0] in ['1', '2']:
        teethMesh.apply_flip()
        labeledMesh.apply_flip()
    
    # === find the maxz --> set to 0 === #
    #print(teethMesh.bounds)
    maxz = teethMesh.bounds[1][2]
    zMatrix = transformations.translation_matrix([0.0, 0.0, -maxz])
    
    teethMesh.apply_transform(zMatrix)
    labeledMesh.apply_transform(zMatrix)
    
    mesh_samples = np.zeros((4*len(labeledMesh.triangles),3))
    for i, f in enumerate(labeledMesh.triangles):
        _mean = np.mean(f, axis=0)
        mesh_samples[i*4:(i*4+3), :] = f
        mesh_samples[(i*4+3), :] = _mean
    
    fs, fnorm = [], []
    crowns, gums = [], []
    for i, f in enumerate(teethMesh.triangles):
        _mean = np.mean(f, axis=0)
        diff = mesh_samples - _mean
        dist = LA.norm(diff, axis=1)
        min_dist = np.amin(dist)
    
        if min_dist < 0.2:
            fs.append(f)
            fnorm.append(teethMesh.face_normals[i])
            #print(f)
            for v in f:
                vertex = " ".join(["{:.6f}".format(p) for p in v])
                crowns.append(vertex)
                
            _mean = np.mean(f, axis=0)
            vertex = " ".join(["{:.6f}".format(p) for p in _mean])
            crowns.append(vertex)
            #print(crowns)
            #sys.exit(1)
        else:
            for v in f:
                vertex = " ".join(["{:.6f}".format(p) for p in v])
                gums.append(vertex)
                
            _mean = np.mean(f, axis=0)
            vertex = " ".join(["{:.6f}".format(p) for p in _mean])
            gums.append(vertex)
    
    with open(os.path.join(label_folder, "t{}_crowns.txt".format(tid)), "w") as fid:
        fid.write("\n".join(crowns))

    with open(os.path.join(label_folder, "t{}_gums.txt".format(tid)), "w") as fid:
        fid.write("\n".join(gums))
    

if __name__ == '__main__':
    cur_folder = os.getcwd()
    data_folder = os.path.join(cur_folder, "data")
    label_folder = os.path.join(cur_folder, "label")
    
    if not os.path.isdir(label_folder):
        os.makedirs(label_folder)
    
    for root, dirs, files in os.walk(data_folder):
        for f in files:
            par_folder = os.path.basename(root)
            test = re.match(r"^t([0-9]+).cut.stl", f)
            if test:
                tid = test.group(1)
                
                output_folder = os.path.join(label_folder, par_folder)
                
                conf_path = os.path.join(root, "teeth.edgeRefined.conf")                
                tcut_path = os.path.join(root, f)
                texport_path = os.path.join(root, "t{}.export.stl".format(tid))
                if os.path.isfile(texport_path):
                    print(tid, conf_path, tcut_path, texport_path, output_folder)
                    teethmesh_labeling(tid, conf_path, tcut_path, texport_path, output_folder)
                else:
                    pass
                    



"""
INFOCONF = "./test/teeth.edgeRefined.conf"
tid = 41

file_path = "./test/t{}.cut.stl".format(tid)
fobj_teeth = open(file_path, "rb")
stlTeeth = stl.load_stl(fobj_teeth)

teethMesh = mesh.aaaMesh(vertices=stlTeeth['vertices'],
                         faces=stlTeeth['faces'],
                         face_normals=stlTeeth['face_normals'])

centroid, rotateDegrees, centerDiff = meshAlign(INFOCONF, tid)
tMatrix = transformations.translation_matrix([-centroid[0], -centroid[1], -centroid[2]])
teethMesh.apply_transform(tMatrix)

for r in rotateDegrees:
    rMatrix = transformations.rotation_matrix(r[0], r[1])
    teethMesh.apply_transform(rMatrix)

if str(tid)[0] in ['1', '2']:
    teethMesh.apply_flip()

# === find the maxz --> set to 0 === #
print(teethMesh.bounds)
maxz = teethMesh.bounds[1][2]
zMatrix = transformations.translation_matrix([0.0, 0.0, -maxz])
teethMesh.apply_transform(zMatrix)
print(teethMesh.bounds)

stl.export_mesh(teethMesh, 'rotated.stl')


label_path = "./test/t{}.export.stl".format(tid)
fobj_teeth = open(label_path, "rb")
labelTeeth = stl.load_stl(fobj_teeth)

labeledMesh = mesh.aaaMesh(vertices=labelTeeth['vertices'],
                           faces=labelTeeth['faces'],
                           face_normals=labelTeeth['face_normals'])

for r in rotateDegrees:
    rMatrix = transformations.rotation_matrix(r[0], r[1])
    labeledMesh.apply_transform(rMatrix)

if str(tid)[0] in ['1', '2']:
    labeledMesh.apply_flip()

labeledMesh.apply_transform(zMatrix)

mesh_samples = np.zeros((4*len(labeledMesh.triangles),3))
for i, f in enumerate(labeledMesh.triangles):
    _mean = np.mean(f, axis=0)
    mesh_samples[i*4:(i*4+3), :] = f
    mesh_samples[(i*4+3), :] = _mean

fs, fnorm = [], []
crowns, gums = [], []
for i, f in enumerate(teethMesh.triangles):
    _mean = np.mean(f, axis=0)
    diff = mesh_samples - _mean
    dist = LA.norm(diff, axis=1)
    min_dist = np.amin(dist)
    
    if min_dist < 0.2:
        fs.append(f)
        fnorm.append(teethMesh.face_normals[i])
        #print(f)
        for v in f:
            vertex = " ".join(["{:.6f}".format(p) for p in v])
            crowns.append(vertex)
    else:
        for v in f:
            vertex = " ".join(["{:.6f}".format(p) for p in v])
            gums.append(vertex)
    

extractMesh = mesh.aaaMesh(vertices=np.array(fs).reshape((-1, 3)),
                           faces=np.arange(len(fs)*3).reshape((-1, 3)),
                           face_normals=fnorm)
stl.export_mesh(extractMesh, 'extracted.stl')

with open("t{}_crowns.txt".format(tid), "w") as fid:
    fid.write("\n".join(crowns))

with open("t{}_gums.txt".format(tid), "w") as fid:
    fid.write("\n".join(gums))
"""