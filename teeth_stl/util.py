# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:27:55 2018

@author: lynden
"""
import numpy as np

import copy
import hashlib
import os
import zipfile

from networkx import Graph, DiGraph, bfs_edges

from sys import version_info

# a flag to check for Python 3
PY3 = version_info.major >= 3

# include constants here so we don't have to import
# a floating point threshold for 0.0
# we are setting it to 100x the resolution of a float64
# which works out to be 1e-13
TOL_ZERO = np.finfo(np.float64).resolution * 100


class PointCloud(object):
    def __init__(self, *args, **kwargs):
        pass


def angle(u, v):
    """
    Returns the angle between vector u, v
    Returns
    --------
    angles: float
    """
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    ang = np.arccos(np.clip(np.einsum('i,i->', u, v), -1, 1)) * (180.0/np.pi)
    return ang

def face_angles(face):
    """
    Returns the angle at each vertex of a face.
    Returns
    --------
    angles: (n, 3) float, angle at each vertex of a face.
    """
    
    u = face[1] - face[0]
    v = face[2] - face[0]
    w = face[2] - face[1]
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    w /= np.linalg.norm(w)
    a = np.arccos(np.clip(np.einsum('i,i->', u, v), -1, 1)) * (180.0/np.pi)
    b = np.arccos(np.clip(np.einsum('i,i->', -u, w), -1, 1)) * (180.0/np.pi)
    #c = np.pi - a - b
    c = 180.0 - a - b

    return [(a, (0, 1, 2)), (b, (1, 0, 2)), (c, (2, 0, 1))]

def unzip(fzip):
    try:
        _zip = zipfile.ZipFile(fzip)
        name = os.path.splitext(os.path.basename(fzip))[0]
        folder = fzip.split(name)[0]
        zipFolder = os.path.join("C:/tmp/zip/", name)
        if not os.path.isdir(zipFolder):
            os.makedirs(zipFolder)
        _zip.extractall(path=zipFolder)
    except zipfile.BadZipfile:
        print("Error: bad zip file {}".format(fzip))
    
    return zipFolder

def zipUpload(folder):
    """
    """
    #print folder
    #print folder.split('/')
    folderName = folder.split('/')[-1]
    zipName = folder.split('/')[-1]+'.zip'
    zipFolder = os.path.join(folder, 'upload/')
    #print zipName
    
    try:
        zipf = zipfile.ZipFile(zipName, 'w', zipfile.ZIP_DEFLATED)
        
        cwd = os.getcwd()
        #print('current folder: {}'.format(cwd))
        os.chdir(zipFolder)
        for root, dirs, files in os.walk(".\\"):
            #print root, dirs, files
            for f in files:
                zipf.write(os.path.join(root, f))
    
        zipf.close()
        print("save folder name: {}".format(folderName))
        # go back to the default folder
        os.chdir(cwd)
        print("go to path: {}".format(cwd))
        if os.path.exists(folderName+'.bter'):
            os.remove(folderName+'.bter')
            print("remove old bter....")
            os.rename(zipName, folderName+'.bter')
        else:
            os.rename(zipName, folderName+'.bter')
    except zipfile.BadZipfile:
        print("Error: folder: {} bad to zip".format(zipFolder))
    except WindowsError as werr:
        print("Error: no files, error message <{}>".format(werr.message))


def convex_hull(points):
    """
    Computes the convex hull of a set of 2D points.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    
    Parameters
    -----------
    points: an iterable sequence of (x, y) pairs representing the points.
    
    Returns
    ----------
    Output: lower-side, upper-side 
      a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    """
    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))
    
    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points
    
    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1], upper[:-1]


def unitize(vectors,
            threshold=None):
    """
    Unitize a vector or an array or row- vectors.
    
    Parameters
    ---------
    vectors : (n,m) or (j) float
       Vector or vectors to be unitized
    
    Returns
    ---------
    unit :  (n,m) or (j) float
       Input vectors but unitized
    """
    # make sure we have a numpy array
    vectors = np.asanyarray(vectors)
    
    # allow user to set zero threshold
    if threshold is None:
        threshold = TOL_ZERO
    
    if len(vectors.shape) == 2:
        # for (m, d) arrays take the per- row unit vector
        # using sqrt and avoiding exponents is slightly faster
        # also dot with ones is faser than .sum(axis=1)
        norm = np.sqrt(np.dot(vectors * vectors,
                              np.ones(vectors.shape[1])))
        # non-zero norms
        valid = norm > threshold
        # in-place reciprocal of nonzero norms
        norm[valid] **= -1
        # tile reciprocal of norm
        tiled = np.tile(norm, (vectors.shape[1], 1)).T
        # multiply by reciprocal of norm
        unit = vectors * tiled
    elif len(vectors.shape) == 1:
        # treat 1D arrays as a single vector
        norm = np.sqrt((vectors * vectors).sum())
        valid = norm > threshold
        if valid:
            unit = vectors / norm
        else:
            unit = vectors.copy()
    else:
        raise ValueError('vectors must be (n, ) or (n, d)!')
    
    return unit
    
def connectivity(dig):
    """
    Parameters
    -----------
    
    Returns
    -----------
    """
    connectedList = []
    
    G = Graph(dig)
    diGraph = DiGraph(G)  # directed graph
    
    restIdx = sorted(dig.keys())
    while len(restIdx) > 0:
        selectId = restIdx[0]
        tmpList = []
        for bfs in bfs_edges(diGraph, selectId):
            for i in xrange(2):
                tmpList.append(bfs[0])
                tmpList.append(bfs[1])
        tmpList = sorted(list(set(tmpList)))
        connectedList.append(copy.copy(tmpList))
        # update the rest index
        restIdx = [item for item in restIdx if item not in set(tmpList)]
        
    connectedList = sorted([(len(i), i) for i in connectedList])
    
    return connectedList

def boundaryMeshConnectivity(ring, faces):
    """
    Parameters
    ------------
    
    Returns
    ------------
    """
    DiG = {}
    
    def vxCheck(DiG, tarId, id1, id2, vx1, vx2):
        edges = ring.fromVidGetEdges(tarId)
        v1count, v2count = 0, 0
        for e in edges:
            if np.array_equal(e[0], vx1):
                v1count = e[1]
            if np.array_equal(e[0], vx2):
                v2count = e[1]
            
        if ring.isVidBoundary(tarId):
            if tarId not in DiG.keys():
                DiG[tarId] = []
            if ring.isVidBoundary(id1) and v1count == 1:
                DiG[tarId].append(id1)    # connect tarId and id1
                if id1 not in DiG.keys():
                    DiG[id1] = []
                DiG[id1].append(tarId)    # connect id1 and tarId
            if ring.isVidBoundary(id2) and v2count == 1:
                DiG[tarId].append(id2)    # connect tarId and id2
                if id2 not in DiG.keys():
                    DiG[id2] = []
                DiG[id2].append(tarId)    # connect id2 and tarId
    
    # === construct directed graph for each input face ====
    for i in xrange(len(faces)):
        vid1 = ring.fromVx2Vid(faces[i][0])
        vid2 = ring.fromVx2Vid(faces[i][1])
        vid3 = ring.fromVx2Vid(faces[i][2])
        
        vxCheck(DiG, vid1, vid2, vid3, faces[i][1], faces[i][2])
        vxCheck(DiG, vid2, vid1, vid3, faces[i][0], faces[i][2])
        vxCheck(DiG, vid3, vid1, vid2, faces[i][0], faces[i][1])
    
    # === remove the duplicated index ===
    for i in DiG.keys():
        DiG[i] = list(set(DiG[i]))
    
    #print 'DiG: ', DiG
    connectedList = connectivity(DiG)
    #print 'connectedList: ', connectedList
    return connectedList, DiG

def is_sequence(obj):
    """
    Check if an object is a sequence or not.
    Parameters
    -------------
    obj : object
      Any object type to be checked
    Returns
    -------------
    is_sequence : bool
        True if object is sequence
    """
    seq = (not hasattr(obj, "strip") and
           hasattr(obj, "__getitem__") or
           hasattr(obj, "__iter__"))
    
    # check to make sure it is not a set, string, or dictionary
    seq = seq and all(not isinstance(obj, i) for i in (dict,
                                                       set,
                                                       basestring))
    
    # Other check criteria
    # PointCloud objects can look like an array but are not
    seq = seq and type(obj).__name__ not in ['PointCloud']
    
    # numpy sometimes returns objects that are single float64 values
    # but sure look like sequences, so we check the shape
    if hasattr(obj, 'shape'):
        seq = seq and obj.shape != ()
        
    return seq

def is_shape(obj, shape):
    """
    Compare the shape of a numpy.ndarray to a target shape,
    with any value less than zero being considered a wildcard
    Note that if a list- like object is passed that is not a numpy
    array, this function will not convert it and will return False.
    Parameters
    ---------
    obj :   np.ndarray
       Array to check the shape on
    shape : list or tuple
       Any negative term will be considered a wildcard
       Any tuple term will be evaluated as an OR
    Returns
    ---------
    shape_ok: bool, True if shape of obj matches query shape
    Examples
    ------------------------
    In [1]: a = np.random.random((100, 3))
    In [2]: a.shape
    Out[2]: (100, 3)
    In [3]: trimesh.util.is_shape(a, (-1, 3))
    Out[3]: True
    In [4]: trimesh.util.is_shape(a, (-1, 3, 5))
    Out[4]: False
    In [5]: trimesh.util.is_shape(a, (100, -1))
    Out[5]: True
    In [6]: trimesh.util.is_shape(a, (-1, (3, 4)))
    Out[6]: True
    In [7]: trimesh.util.is_shape(a, (-1, (4, 5)))
    Out[7]: False
    """
    
    # if the obj.shape is different length than
    # the goal shape it means they have different number
    # of dimensions and thus the obj is not the query shape
    if (not hasattr(obj, 'shape') or
            len(obj.shape) != len(shape)):
        return False
    
    # loop through each integer of the two shapes
    # multiple values are sequences
    # wildcards are less than zero (i.e. -1)
    # ex: zip((100, 3), (-1, 3)) --> [(100, -1), (3, 3)]
    for i, target in zip(obj.shape, shape):
        # check if current field has multiple acceptable values
        if is_sequence(target):
            if i in target:
                # obj shape is in the accepted values 
                continue
            else:
                return False
        # check if current field is a wildcard
        if target < 0:
            if i == 0:
                # if a dimension is 0, we don't allow
                # that to match to a wildcard
                # it would have to be explicitly called out as 0
                return False
            else:
                continue        
        # since we have a single target and a single value,
        # if they are not equal we have an answer
        if target != i:
            return False    
    
    # since none of the checks failed the obj.shape
    # matches the pattern
    return True

def md5_object(obj):
    """
    If an object is hashable, return the string of the MD5.
    Parameters
    -----------
    obj: object
    Returns
    ----------
    md5: str, MD5 hash
    """
    
    hasher = hashlib.md5()
    if isinstance(obj, basestring) and PY3:
        hasher.update(obj.encode('utf-8'))
    else:
        hasher.update(obj)

    md5 = hasher.hexdigest()
    return md5

def md5_array(array, digits=5):
    """
    Parameters
    ---------
    array:  numpy array
    digits: int, number of digits to account for in the MD5
    Returns
    ---------
    md5: str, md5 hash of input
    """
    digits = int(digits)
    array = np.asanyarray(array, dtype=np.float64).reshape(-1)
    as_int = (array * 10 ** digits).astype(np.int64)
    md5 = md5_object(as_int.tostring(order='C'))
    return md5

'''
def bounds_tree(bounds):
    """
    Given a set of axis aligned bounds, create an r-tree for broad-phase
    collision detection
    Parameters
    ---------
    bounds: (n, dimension*2) list of interleaved bounds
             for a 2D bounds tree: (interleaved default is True)
             [(xmin, ymin, zmin, xmax, ymax, zmax), ...]
    Returns
    ---------
    tree: Rtree object
    """
    bounds = np.asanyarray(copy.deepcopy(bounds), dtype=np.float64)
    if len(bounds.shape) != 2:
        raise ValueError('Bounds must be (n,dimension*2)!')
    
    dimension = bounds.shape[1]
    if (dimension % 2) != 0:
        raise ValueError('Bounds must be (n,dimension*2)!')
    dimension = int(dimension / 2)
    
    import rtree
    # some versions of rtree screw up indexes on stream loading
    # do a test here so we know if we are free to use stream loading
    # or if we have to do a loop to insert things which is 5x slower
    rtree_test = rtree.index.Index([(1564, [0, 0, 0, 10, 10, 10], None)],
                                   properties=rtree.index.Property(dimension=3))
    rtree_stream_ok = next(rtree_test.intersection([1, 1, 1, 2, 2, 2])) == 1564
    
    properties = rtree.index.Property(dimension=dimension)
    if rtree_stream_ok:
        tree = rtree.index.Index(zip(np.arange(len(bounds)),
                                     bounds,
                                     [None] * len(bounds)),
                                 properties=properties)
    else:
        tree = rtree.index.Index(properties=properties)
        for i, b in enumerate(bounds):
            tree.insert(i, b)
    return tree
'''

"""
pc = PointCloud()
print type(pc)
print type(pc).__name__

md5 = md5_object('123')
md5 = md5_array(np.array([[[1, 2, 3], [3, 3, 4], [4, 5, 6]],[[1, 1, 1], [2, 2, 2], [4,3,1]]]))
print md5


import rtree
# some versions of rtree screw up indexes on stream loading
# do a test here so we know if we are free to use stream loading
# or if we have to do a loop to insert things which is 5x slower
rtree_test = rtree.index.Index([(1564, [0, 0, 0, 10, 10, 10], None)],
                               properties=rtree.index.Property(dimension=3))
rtree_stream_ok = next(rtree_test.intersection([1, 1, 1, 2, 2, 2])) == 1564

print 'rtree = ', rtree_test
print rtree_test.intersection([1, 1, 1, 2, 2, 2])
print rtree_stream_ok
"""