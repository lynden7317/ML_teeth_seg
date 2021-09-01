# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 16:17:49 2018

@author: lynden
"""

import math
import numpy as np


def unit_vector(data):
    """
    Return ndarray normalized by length
    """
    data = np.array(data, dtype=np.float64, copy=True)
    data /= math.sqrt(np.dot(data, data))
    return data

def identity_matrix():
    return np.identity(4)

def translation_matrix(direction):
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M

def rotation_matrix(angle, direction):
    """
    Return matrix to rotate about axis defined by direction
    
    Parameters
    ----------------
    angle:        float in radians
    direction:    (3,) float, unit vector along rotation axis
    
    Returns
    ----------------
    matrix:       (4, 4) float homogenous transformation matrix
    
    ex: 4x4 matrix form (rotate in x-axis)
    M = [[1, 0, 0, 0], [0, cosa, -sina, 0], [0 sina, cosa, 0], [0, 0, 0, 1]]
    """
    
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    M = np.diag([cosa, cosa, cosa, 1.0])
    M[:3, :3] += np.outer(direction, direction) * (1.0 - cosa)
    
    direction = direction * sina
    M[:3, :3] += np.array([[0.0, -direction[2], direction[1]],
                           [direction[2], 0.0, -direction[0]],
                           [-direction[1], direction[0], 0.0]])
    
    return M
    
def transform_points(points,
                     matrix,
                     translate=True):
    """
    Returns points, rotated by transformation matrix
    If points is (n,2), matrix must be (3,3)
    if points is (n,3), matrix must be (4,4)
    Parameters
    ----------
    points    : (n, d) float
                  Points where d is 2 or 3
    matrix    : (3,3) or (4,4) float
                  Homogenous rotation matrix
    
    Returns
    ----------
    transformed : (n,d) float
                   Transformed points
    """
    points = np.asanyarray(points, dtype=np.float64)
    matrix = np.asanyarray(matrix, dtype=np.float64)
    
    dimension = points.shape[1]
    column = np.zeros(len(points)) + int(bool(translate))
    stacked = np.column_stack((points, column))
    transformed = np.dot(matrix, stacked.T).T[:, :dimension]
    transformed = np.ascontiguousarray(transformed)
    return transformed

def isRotationMatrix(R):
    """
    Checks if a matrix is a valid rotation matrix.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    """
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """
 
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