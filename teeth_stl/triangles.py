# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:28:44 2018

@author: A30367
"""
import numpy as np

def cross(triangles):
    """
    Returns the cross product of two edges from input triangles
    Parameters
    --------------
    triangles: (n, 3, 3) float, vertices of triangles
    Returns
    --------------
    crosses: (n, 3) float, cross product of two edge vectors
    """
    vectors = np.diff(triangles, axis=1)
    crosses = np.cross(vectors[:, 0], vectors[:, 1])
    return crosses

def area(triangles=None, crosses=None, sum=False):
    """
    Calculates the sum area of input triangles
    Parameters
    ----------
    triangles: vertices of triangles (n,3,3)
    sum:       bool, return summed area or individual triangle area
    Returns
    ----------
    area:
        if sum: float, sum area of triangles
        else:   (n,) float, individual area of triangles
    """
    if crosses is None:
        crosses = cross(triangles)
    area = (np.sum(crosses**2, axis=1)**.5) * .5
    if sum:
        return np.sum(area)
    return area