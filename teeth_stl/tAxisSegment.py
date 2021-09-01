# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 16:31:21 2017

@author: Lynden
"""

import numpy as np
import math
import json

'''
R = tAxisSegment.rotationMatrix('t'+str(tid), axisConf)
print R
angles = tAxisSegment.rotationMatrixToEulerAngles(R)
XAxis, YAxis, ZAxis = angles[0], angles[1], angles[2]
tooth.reduceGum(['manual', [XAxis, YAxis, ZAxis]])
'''

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def rotationMatrix_treatment(tid, axisConf):
    R = np.array([[axisConf[tid]['axis']['xAxis']['x'], axisConf[tid]['axis']['xAxis']['y'], axisConf[tid]['axis']['xAxis']['z']],
                  [axisConf[tid]['axis']['yAxis']['x'], axisConf[tid]['axis']['yAxis']['y'], axisConf[tid]['axis']['yAxis']['z']],
                  [axisConf[tid]['axis']['zAxis']['x'], axisConf[tid]['axis']['zAxis']['y'], axisConf[tid]['axis']['zAxis']['z']]
                  ])
    
    return R

def rotationMatrix_select(axisConf):
    if 'xAxisVector' in axisConf.keys():
        R = np.array([[axisConf['xAxisVector']['x'], axisConf['xAxisVector']['y'], axisConf['xAxisVector']['z']],
                      [axisConf['yAxisVector']['x'], axisConf['yAxisVector']['y'], axisConf['yAxisVector']['z']],
                      [axisConf['zAxisVector']['x'], axisConf['zAxisVector']['y'], axisConf['zAxisVector']['z']]
                     ])
    else:
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    return R

def centroidShift_select(axisConf):
    if 'axesCenterDiff' in axisConf.keys():
        return [axisConf['axesCenterDiff']['x'], axisConf['axesCenterDiff']['y'], axisConf['axesCenterDiff']['z']]
    else:
        return [0.0, 0.0, 0.0]

# Deprecated
def readAxis(outputFolder):
    axisConf = {}
    # >>>> load axis
    try:
        with open(outputFolder + 'axis.json') as fin:
            axisConf = json.load(fin)
    except IOError:
        print 'Read Axis error'
    
    return axisConf

