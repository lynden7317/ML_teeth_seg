# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 23:55:14 2018

@author: lynden
"""

import os
import json
import numpy as np

from teeth_stl import transformations
from teeth_stl import stl
from teeth_stl import mesh


def selectCentroidShift(axisConf):
    """
    Parameter
    ----------
    axisConf: json format input
    
    Return
    ----------
    list of values
    """
    if 'axesCenterDiff' in axisConf.keys():
        return [axisConf['axesCenterDiff']['x'], axisConf['axesCenterDiff']['y'], axisConf['axesCenterDiff']['z']]
    else:
        return [0.0, 0.0, 0.0]

def selectRotationMatrix(axisConf):
    """
    Parameter
    ----------
    axisConf: json format input
    
    Return
    ----------
    R: (3, 3) rotation matrix
    """
    if 'xAxisVector' in axisConf.keys():
        R = np.array([[axisConf['xAxisVector']['x'], axisConf['xAxisVector']['y'], axisConf['xAxisVector']['z']],
                      [axisConf['yAxisVector']['x'], axisConf['yAxisVector']['y'], axisConf['yAxisVector']['z']],
                      [axisConf['zAxisVector']['x'], axisConf['zAxisVector']['y'], axisConf['zAxisVector']['z']]
                     ])
    else:
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    return R

def loadJsonConf(folderPath, jsonName):
    path = os.path.join(folderPath, jsonName)
    try:
        with open(path) as fin:
            loadJson = json.load(fin)
    except IOError:
        print("ERROR <loadJsonConf>: %s not find" %path)
    
    return loadJson


# ==== tester ====
"""
folderPath = '.\conf'
jsonName = 'teeth.conf'
loadJson = loadJsonConf(folderPath, jsonName)
print loadJson

file_path = './conf/t33.reduceCut.stl'

file_obj = open(file_path, 'rb')
stlmesh = stl.load_stl(file_obj)            
aaaMesh = mesh.aaaMesh(vertices=stlmesh['vertices'], 
                       faces=stlmesh['faces'], 
                       face_normals=stlmesh['face_normals'])

for key, values in loadJson.items():
    print key
    print values
    R = selectRotationMatrix(values)
    angles = transformations.rotationMatrixToEulerAngles(R)
    rotateDegrees = [(angles[0], [1,0,0]), (angles[1], [0,1,0]), (angles[2], [0,0,1])]
    print rotateDegrees
    
    if key == 't33':
        for r in rotateDegrees:
            rMatrix = transformations.rotation_matrix(r[0], r[1])
            aaaMesh.apply_transform(rMatrix)
        stl.export_mesh(aaaMesh, './conf/meshRotate.stl')
"""