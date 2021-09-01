# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:26:52 2018

@author: lynden
"""
import numpy as np

import copy
import time
import itertools as it

from networkx import Graph, DiGraph, bfs_edges

# = defined packages =
from teeth_stl import stl
from teeth_stl import mesh
from teeth_stl import util
# ====================

def vxRound(v, decimals=3):
    return np.around(v-10**(-(decimals+5)), decimals=decimals)

class vertexStore(object):
    """
    A class to store numpy array vector with hash function
    the key is vertex with numpy array type
    the data is vertex index, the valid index should >0
    -------------------------
    vxStore = vertexStore()
    vxStore[np.array([1, 1, 1])] = 0
    print vxStore[np.array([1, 1, 1])]  --> print 0
    """
    def __init__(self, bucket=512):
        super(vertexStore, self).__init__()
        self.bucket = bucket
        self.len = 0
        self.currentId = 0
        self.hashmap = [[] for i in range(self.bucket)]
    
    @property
    def vid(self):
        return self.currentId
    
    def is_empty(self):
        """
        Is the current vertexStore empty or not
        
        Return
        ----------
        empty: bool, False if there are items in the vertexStore
        """
        if len(self.hashmap) == 0:
            return True
        if sum([len(v) for v in self.hashmap]) == 0:
            return True
        return False
    
    def clear(self):
        """
        Remove all data from the vertexStore
        """
        self.len = 0
        self.hashmap = []
    
    def removeVx(self, vertex):
        """
        """
        hash_key = self._hash(vertex) % self.bucket
        bucket = self.hashmap[hash_key]
        popx = None
        for i, kv in enumerate(bucket):
            k, v = kv
            if np.array_equal(vertex, k):
                popx = i
                break
        if popx is not None:
            #print 'remove at: ', (hash_key, popx)
            self.hashmap[hash_key].pop(popx)
            self.len -= 1
    
    def __getitem__(self, vertex):
        """
        Return
        ----------
        -1: no item find in the vertexStore
        """
        hash_key = self._hash(vertex) % self.bucket
        bucket = self.hashmap[hash_key]
        for i, kv in enumerate(bucket):
            k, v = kv
            if np.array_equal(vertex, k):  #np.allclose
                return v
        return -1
    
    def __setitem__(self, vertex, vid):
        """
        Insert the vertex into the self.hashmap
        If the duplicated vertex was inserted, ignore the duplicated one
        """
        hash_key = self._hash(vertex) % self.bucket
        key_exist = False
        bucket = self.hashmap[hash_key]
        for i, kv in enumerate(bucket):
            k, v = kv
            if np.array_equal(vertex, k):
                key_exist = True
                break
        if key_exist:
            # maintain the vertex with the first insertion
            #bucket[i] = (vertex, vid)
            pass
        else:
            bucket.append((vertex, vid))
            self.len += 1
            self.currentId += 1
    
    def __len__(self):
        return self.len
    
    def _hash(self, vertex):
        array = np.asanyarray(vertex, dtype=np.float64).reshape(-1)
        as_int = (array * 10 ** 5).astype(np.int64)
        return hash(as_int.tostring(order='C'))


class oneRing(object):
    def __init__(self):
        """
        self._data = {vid:{'edge':[], 'face':[], 'vx':vx, 'norm':[], 'seq':[]}, ...}
        """
        super(oneRing, self).__init__()
        self._data = {}
        
        self.vxStore = vertexStore(bucket=512)
    
    def __len__(self):
        return len(self.vxStore)
    
    def _tri(self, vx, vx1, vx2, seq):
        tri = {
                "1, 2": [vx, vx1, vx2],
                "0, 2": [vx1, vx, vx2],
                "0, 1": [vx1, vx2, vx] }[seq]
        return tri
    
    def isVidBoundary(self, vid):
        if len(self._data[vid]['face']) < len(self._data[vid]['edge']):
            return True
        else:
            return False
    
    def fromVidGetEdges(self, vid):
        return self._data[vid]['edge']
    
    def fromVidGetFaces(self, vid):
        return self._data[vid]['face']
    
    def fromVidGetNorms(self, vid):
        return self._data[vid]['norm']
    
    def fromVidGetVx(self, vid):
        return self._data[vid]['vx']
    
    def fromVx2Vid(self, vx):
        return self.vxStore[vx]
    
    def meshIdLoops(self):
        """
        the connectivity of vertex in the mesh
        if it has more than one element, it has more than
        one sub-mesh in the mesh
        
        Return
        --------
        list: [(len(loop), loop=[idx, ....]), ...]
        """
        #>>> DiGraph = { node: [connected_nodes] }
        DiG = {}
        
        for vid in sorted(self._data.keys()):
            DiG[vid] = []
            for e in self._data[vid]['edge']:
                _vid = self.vxStore[e]
                DiG[vid].append(_vid)
        
        # == remove the duplicated index ==
        for vid in DiG.keys():
            DiG[vid] = list(set(DiG[vid]))
    
        #print 'DiG = ', DiG
        #print 'DiG key = ', DiG.keys()
        # == build the connected list ==
        connectedList = util.connectivity(DiG)
        #print 'connectedList: ', connectedList
        return connectedList
    
    def removeFace(self, face):
        vx1, vx2, vx3 = face[0], face[1], face[2]
        meanF = np.mean(face, axis=0)
        meanF = vxRound(meanF)
        #print 'meanF = ', meanF
        vid = [self.vxStore[vx1], self.vxStore[vx2], self.vxStore[vx3]]
        for _vid in vid:
            #print '_vid = ', _vid
            _vx1 = self._data[_vid]['vx']
            popList = []
            for i, f in enumerate(self._data[_vid]['face']):
                _vx2, _vx3 = f[0], f[1]
                seq = self._data[_vid]['seq'][i]
                tri = self._tri(_vx1, _vx2, _vx3, seq)
                meanT = np.mean(tri, axis=0)
                meanT = vxRound(meanT, 3)
                #print 'i, tri = ', i, tri
                #print 'meanT = ', meanT
                if np.array_equal(meanF, meanT):
                    popList.append(i)
            #print 'pop list = ', popList
            # >>> update self._data include 'face', 'edge', 'seq', 'norm' <<<
            for i in reversed(popList):
                e1, e2 = self._data[_vid]['face'][i][0], self._data[_vid]['face'][i][1]
                edgeRemoveList = []
                for ei, _e in enumerate(self._data[_vid]['edge']):
                    if np.array_equal(_e[0], e1):
                        _e[1] -= 1
                    if np.array_equal(_e[0], e2):
                        _e[1] -= 1
                    if _e[1] == 0:
                        edgeRemoveList.append(ei)
                
                # >>> pop the elements in self._data <<<
                for ei in reversed(edgeRemoveList):
                    self._data[_vid]['edge'].pop(ei)
                self._data[_vid]['face'].pop(i)
                self._data[_vid]['seq'].pop(i)
                self._data[_vid]['norm'].pop(i)
            
            # >>> pop the elements in self.vxStore <<<
            if len(self._data[_vid]['face']) == 0:
                self.vxStore.removeVx(_vx1)
                delKey = self._data.pop(_vid, None)
                #print '*** del key in self._data: ', delKey
            
            #print 'update face: ', self._data[_vid]['face']
            #print 'update edge: ', self._data[_vid]['edge']
            #print 'update seq: ', self._data[_vid]['seq']
            #print 'update norm: ', self._data[_vid]['norm']
    
    def removeNonManifoldFaces(self):
        faces = self.nonManifoldFaces
        while len(faces) > 0:
            #print 'removeNonManifold, len(faces)= ', len(faces)
            for f in faces:
                self.removeFace(f)
            
            faces = self.nonManifoldFaces
    
    def removeThreeVxBoundaryFaces(self): #, getRemoveFaces=False):
        totalFaceList, totalFaceNormList = [], []
        while(1):
            faceList, faceNormList = [], []
            bf, bfNorm = self.boundaryFaces
            #print len(boundaryFaces)
            loop, DiG = util.boundaryMeshConnectivity(self, bf)
        
            boundaryLoop = sorted(loop, reverse=True)[0][1]  # the boundary loop has the maximum value
            for i, f in enumerate(bf):
                vxCount = 0
                if self.vxStore[f[0]] in boundaryLoop:
                    vxCount += 1
                if self.vxStore[f[1]] in boundaryLoop:
                    vxCount += 1
                if self.vxStore[f[2]] in boundaryLoop:
                    vxCount += 1
                if vxCount > 2:
                    faceList.append(f)
                    faceNormList.append(bfNorm[i])
                    totalFaceList.append(f)
                    totalFaceNormList.append(bfNorm[i])
        
            for f in faceList:
                self.removeFace(f)
            
            if len(faceList) == 0:
                break
        
        return totalFaceList, totalFaceNormList
        #if getRemoveFaces:
        #    return faceList, faceNormList
        #else:
            # directly remove the face in the faceList
        #    for f in faceList:
        #        self.removeFace(f)
    
    @property
    def stlMesh(self):
        stlMeshList, stlNormList = [], []
        nvStore = vertexStore(bucket=4096)
        for vid in self._data.keys():
            vx = self._data[vid]['vx']
            for fid in xrange(len(self._data[vid]['face'])):
                vx1, vx2 = self._data[vid]['face'][fid][0], self._data[vid]['face'][fid][1]
                seq = self._data[vid]['seq'][fid]
                norm = self._data[vid]['norm'][fid]
                tri = self._tri(vx, vx1, vx2, seq)
                meanV = np.mean(tri, axis=0)
                meanV = vxRound(meanV)
                nv = np.concatenate((norm, meanV), axis=None)
                nid = nvStore[nv]
                if nid == -1:
                    nid = len(nvStore)
                    nvStore[nv] = nid
                    stlMeshList.append(copy.deepcopy(tri))
                    stlNormList.append(copy.copy(norm))
        
        return stlMeshList, stlNormList
    
    @property
    def nonManifoldFaces(self):
        """
        """
        nonManifoldFace = []
        nonManifoldVx, nonManifoldEdge = [], []
        # find the nonManifold edge, #edge > 2
        for vid in self._data.keys():
            for e in self._data[vid]['edge']:
                if e[1] > 2:
                    #print 'vid, vx = ', vid, self._data[vid]['vx']
                    #print '***\n', self._data[vid]['edge']
                    nonManifoldVx.append((vid, self._data[vid]['vx']))
                    break
        
        #print 'nonManifoldVx= ', nonManifoldVx
        # generate the all combination of edges in nonManifoldVx
        tmpEdge = list(it.combinations(nonManifoldVx, 2)) 
        #print 'tmpEdge= ', tmpEdge
        for e in tmpEdge:
            tarId, vxCheck = e[0][0], e[1][1]
            for _e in self._data[tarId]['edge']:
                if np.array_equal(_e[0], vxCheck):
                    nonManifoldEdge.append(e)
                    break
        
        #print '**nonManifoldEdge= ', nonManifoldEdge
        _vxStore = vertexStore(bucket=64)
        for e in nonManifoldEdge:
            tarId, vx = e[0][0], e[1][1]
            candidateVx = []
            for f in self._data[tarId]['face']:
                if np.array_equal(f[0], vx):
                    if _vxStore[f[1]] == -1:
                        _vid = _vxStore.currentId
                        _vxStore[f[1]] = _vid
                        
                        vid = self.vxStore[f[1]]
                        lenEdge = len(self._data[vid]['edge'])
                        candidateVx.append((lenEdge, vid, f[1]))
                        
                if np.array_equal(f[1], vx):
                    if _vxStore[f[0]] == -1:
                        _vid = _vxStore.currentId
                        _vxStore[f[0]] = _vid
                        
                        vid = self.vxStore[f[0]]
                        lenEdge = len(self._data[vid]['edge'])
                        candidateVx.append((lenEdge, vid, f[0]))
            
            candidateVx = sorted(candidateVx)
            fv = candidateVx[0][2]
            nonManifoldFace.append([fv, e[0][1], e[1][1]])
            
        return nonManifoldFace
        #for vx in candidateVx:
        #    vid = self.vxStore[vx]
        #    print 'vx, vid, len(edge)= ', vx, vid, len(self._data[vid]['edge'])
        """
        if len(self._data[vid]['face']) > len(self._data[vid]['edge']):
            print '==== nonManifold ====\n', self._data[vid]['face']
            print '==norm==\n', self._data[vid]['norm']
            print 'seq: ', self._data[vid]['seq']
            print 'vx: ', self._data[vid]['vx']
            print 'vid: ', vid
            print 'edge: ', self._data[vid]['edge']
        """
    
    @property
    def boundaryFaces(self):
        faceList, faceNormList = [], []
        nvStore = vertexStore(bucket=512)
        for vid in self._data.keys():
            if len(self._data[vid]['face']) < len(self._data[vid]['edge']):                
                oneEdge = [_e[0] for _e in self._data[vid]['edge'] if _e[1] == 1]
                oneFaceIds = []
                for _e in oneEdge:
                    for _fid in xrange(len(self._data[vid]['face'])):
                        fv1 = self._data[vid]['face'][_fid][0]
                        fv2 = self._data[vid]['face'][_fid][1]
                        if np.array_equal(_e, fv1):
                            fid = _fid
                        if np.array_equal(_e, fv2):
                            fid = _fid
                    oneFaceIds.append(fid)                    
                    
                for fid in oneFaceIds:
                    vx = self._data[vid]['vx']
                    vx1, vx2 = self._data[vid]['face'][fid][0], self._data[vid]['face'][fid][1]
                    seq = self._data[vid]['seq'][fid]
                    norm = self._data[vid]['norm'][fid]
                    tri = self._tri(vx, vx1, vx2, seq)
                    meanV = np.mean(tri, axis=0)
                    meanV = vxRound(meanV)
                    nv = np.concatenate((norm, meanV), axis=None)
                    nid = nvStore[nv]
                    if nid == -1:
                        #nid = len(nvStore)
                        nid = nvStore.currentId
                        nvStore[nv] = nid
                        faceList.append(copy.deepcopy(tri))
                        faceNormList.append(copy.copy(norm))
                #print '***** e *****: ', self._data[vid]['edge']
                #print oneSideEdge
                
                """
                edgeCount = []
                for _e in self._data[vid]['edge']:
                    _vx = _e[0]
                    count, fid = 0, 0
                    for _fid in xrange(len(self._data[vid]['face'])):
                        fv1 = self._data[vid]['face'][_fid][0]
                        fv2 = self._data[vid]['face'][_fid][1]
                        if np.array_equal(_vx, fv1):
                            count += 1
                            fid = _fid
                        if np.array_equal(_vx, fv2):
                            count += 1
                            fid = _fid
                    edgeCount.append((count, fid))
                
                for num, fid in edgeCount:
                    if num == 1:
                        vx = self._data[vid]['vx']
                        vx1, vx2 = self._data[vid]['face'][fid][0], self._data[vid]['face'][fid][1]
                        seq = self._data[vid]['seq'][fid]
                        norm = self._data[vid]['norm'][fid]
                        tri = self._tri(vx, vx1, vx2, seq)
                        #tri = [vx, vx1, vx2]
                        #vec1, vec2 = vxRound(vx1-vx), vxRound(vx2-vx)
                        meanV = np.mean(tri, axis=0)
                        meanV = vxRound(meanV)
                        #norm = np.cross(vec1, vec2)
                        #unitN = norm/np.linalg.norm(norm)
                        #unitN = vxRound(unitN)
                        nv = np.concatenate((norm, meanV), axis=None)
                        nid = nvStore[nv]
                        if nid == -1:
                            nid = len(nvStore)
                            nvStore[nv] = nid
                            faceList.append(copy.deepcopy(tri))
                            faceNormList.append(copy.copy(norm))
                """
        return faceList, faceNormList                        
    
    def addFace(self, face, normal):
        """
        Parameters
        ----------------------
        face:        (3, 3) float set of vertex locations
        normal:      (1, 3) float set of face normal
        
        Data structure
        ----------------------
        self._data = {value: {'vx'  : (1, 3) float set, 
                              'edge': [ [(1, 3), count], ...] list of (1, 3) float set,
                              'face': [ [(1, 3), (1, 3)], ...] list of two (1, 3) float set,
                              'norm': [(1, 3), ...] list of (1, 3) float set,
                              'seq':  [(1,2)/(0,2)/(0,1), ...] list of (1,2)/(0,2)/(0,1) tuple }}
        """
        for fid in xrange(3):
            vx = face[fid]
            vid = self.vxStore[vx]
            if vid == -1:
                # insert the new vertex
                vid = self.vxStore.currentId
                self.vxStore[vx] = vid
                self._data.update({vid:{'edge':[], 'face':[], 'vx':vx, 'norm':[], 'seq':[]}})
                if fid == 0:
                    self._addface(vid, face[1], face[2], vx, normal, "1, 2")
                    self._data[vid]['edge'].append([face[1].copy(), 1])
                    self._data[vid]['edge'].append([face[2].copy(), 1])
                if fid == 1:
                    self._addface(vid, face[0], face[2], vx, normal, "0, 2")
                    self._data[vid]['edge'].append([face[0].copy(), 1])
                    self._data[vid]['edge'].append([face[2].copy(), 1])
                if fid == 2:
                    self._addface(vid, face[0], face[1], vx, normal, "0, 1")
                    self._data[vid]['edge'].append([face[0].copy(), 1])
                    self._data[vid]['edge'].append([face[1].copy(), 1])
            else:
                # vertex is already in the self.vxStore
                if fid == 0:
                    self._addface(vid, face[1], face[2], vx, normal, "1, 2")
                    self._addedge(vid, face[1], face[2])
                if fid == 1:
                    self._addface(vid, face[0], face[2], vx, normal, "0, 2")
                    self._addedge(vid, face[0], face[2])
                if fid == 2:
                    self._addface(vid, face[0], face[1], vx, normal, "0, 1")
                    self._addedge(vid, face[0], face[1])
    
    def _addface(self, vid, vx1, vx2, vx0, normal, seq):
        self._data[vid]['face'].append([vx1.copy(), vx2.copy()])
        self._data[vid]['norm'].append(normal)
        self._data[vid]['seq'].append(seq)
    
    def _addedge(self, vid, vx1, vx2):
        newV1, newV2 = True, True
        for e in self._data[vid]['edge']:
            _v = e[0]
            if np.array_equal(_v, vx1):
                newV1 = False
                e[1] += 1
            if np.array_equal(_v, vx2):
                newV2 = False
                e[1] += 1

        if newV1:
            self._data[vid]['edge'].append([vx1.copy(), 1])
        if newV2:
            self._data[vid]['edge'].append([vx2.copy(), 1])


"""
start_time = time.time()
vertices = vertexStore()
vx = np.random.randn(2000, 3)
print vx

for i, v in enumerate(vx):
    vertices[v] = i

duration = time.time() - start_time
print('insert 100000 spending = %.3f' %(duration))

#for i in vertices.hashmap:
#    print len(i)

#vertices.is_empty()

start_time = time.time()
print 'find vx[0] id: ', vertices[vx[100]]
duration = time.time() - start_time
print('find vertex id spending = %.3f' %(duration))

print vertices[np.array([1, 1, 1])]
"""


# tester
"""
file_path = './tri/testFillt31.stl'
file_obj = open(file_path, 'rb')

stlmesh = stl.load_stl(file_obj)

aaaMesh = mesh.aaaMesh(vertices=stlmesh['vertices'], 
                       faces=stlmesh['faces'], 
                       face_normals=stlmesh['face_normals'])

#stl.export_mesh(aaaMesh, './tri/test.stl')

start_time = time.time()
meshRing = oneRing()

for i, f in enumerate(aaaMesh.triangles):    
    normal = aaaMesh.face_normals[i]
    #print 'normal = ', normal, type(normal)
    meshRing.addFace(f, normal)

duration = time.time() - start_time
print('struct 1-ring = %.3f sec.' %(duration))
print 'len(meshRing) = ', len(meshRing)

meshRing.removeThreeVxBoundaryFaces()
stlFaces, stlNorms = meshRing.stlMesh

faces, _ = meshRing.removeThreeVxBoundaryFaces(getRemoveFaces=True)
print 'len(faces)= ', len(faces)

mesh = mesh.aaaMesh(vertices=np.array(stlFaces).reshape((-1, 3)),
                      faces=np.arange(len(stlFaces)*3).reshape((-1, 3)),
                      face_normals=stlNorms)

stl.export_mesh(mesh, './tri/removeThreeVxFaces.stl')
"""


# tester
"""
file_path = './tri/testNonmanifold.stl' #'./tri/testT31.hole.stl' #'./tri/testNonmanifold.stl'
#'./tri/t31.cut.stl' #'./tri/Step-6-Att-Mandible.stl'  
file_obj = open(file_path, 'rb')

stlmesh = stl.load_stl(file_obj)
#print stlmesh['vertices'], len(stlmesh['vertices'])
#print stlmesh['face_normals'], len(stlmesh['face_normals'])
            
aaaMesh = mesh.aaaMesh(vertices=stlmesh['vertices'], 
                       faces=stlmesh['faces'], 
                       face_normals=stlmesh['face_normals'])

#stl.export_mesh(aaaMesh, './tri/test.stl')

start_time = time.time()
meshRing = oneRing()

testFace = None
for i, f in enumerate(aaaMesh.triangles):
    if np.allclose(f[0], np.array([-2.35358047,  4.21580124,  2.54877996])):
        testFace = f
        print '***count***: ', f
        print i
    
    normal = aaaMesh.face_normals[i]
    #print 'normal = ', normal, type(normal)
    meshRing.addFace(f, normal)

duration = time.time() - start_time
print('struct 1-ring = %.3f sec.' %(duration))

print 'len(meshRing) = ', len(meshRing)


meshRing.nonManifoldFaces

meshRing.removeNonManifoldFaces()
stlFaces, stlNorms = meshRing.stlMesh
mesh = mesh.aaaMesh(vertices=np.array(stlFaces).reshape((-1, 3)),
                      faces=np.arange(len(stlFaces)*3).reshape((-1, 3)),
                      face_normals=stlNorms)

stl.export_mesh(mesh, './tri/nonManifold.stl')
"""

"""
meshRing.removeFace(testFace)

# ==== test-save to stl ====
stlFaces, stlNorms = meshRing.stlMesh
mesh = mesh.aaaMesh(vertices=np.array(stlFaces).reshape((-1, 3)),
                      faces=np.arange(len(stlFaces)*3).reshape((-1, 3)),
                      face_normals=stlNorms)

stl.export_mesh(mesh, './tri/ringMesh.stl')
# ===========================

print meshRing._data.keys()
print meshRing.vxStore[np.array([-2.35358047,  4.21580124,  2.54877996])]
print 'property: ', meshRing.fromVx2Vid(np.array([-2.35358047,  4.21580124,  2.54877996]))
print 'len of vxStore= ', len(meshRing.vxStore)
print 'currentId of vxStore= ', meshRing.vxStore.currentId
"""

"""
start_time = time.time()
bf, bfNorm = meshRing.boundaryFaces
print len(bf), len(bfNorm)
duration = time.time() - start_time
print('struct boundary = %.3f sec.' %(duration))

bfMesh = mesh.aaaMesh(vertices=np.array(bf).reshape((-1, 3)),
                      faces=np.arange(len(bf)*3).reshape((-1, 3)),
                      face_normals=bfNorm)

stl.export_mesh(bfMesh, './tri/boundary.stl')

loop, DiG = util.boundaryMeshConnectivity(meshRing, bf)
print 'loop: ', loop
"""