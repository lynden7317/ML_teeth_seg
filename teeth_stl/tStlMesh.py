# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 17:31:37 2017

@author: lynden
"""

import logging
import numpy as np
import itertools
import copy
import math
import time
import subprocess
import os

from utils import getEnvVar, withSystemSuffix
# python numpy-stl
from stl import mesh
from networkx import Graph, DiGraph, bfs_edges

tStlTestMode = False

if tStlTestMode:
    import matplotlib.pyplot as plt

# ==== stl rounding function ====
def stlRound(v, decimals=3):
    return np.around(v-10**(-(decimals+5)), decimals=decimals)

# ===== STL preprocessing ====
def preSTLNormal(tmesh):
    # rotation matrix
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    meshVectors = np.dot(tmesh.vectors, R)

    # get normals
    meshNormals = np.cross(meshVectors[::,1]-meshVectors[::,0], meshVectors[::,2]-meshVectors[::,0])

    # construct the output mesh
    _mesh = np.zeros(len(meshVectors), dtype = mesh.Mesh.dtype)
    _mesh['vectors'] = meshVectors
    _mesh['normals'] = meshNormals
    return _mesh

def simplification(inputFile, outputFile, script = 'Simplification_Target_60000_Face'):
    """
    Run meshlab script to simplify mesh

    Args:
        inputFile: ./uploads/plx9qevdmda/UpperJaw.stl
        outputFile: ./uploads/plx9qevdmda/UpperJaw.simplified.stl
        script: path to the script file
    """

    # get meshlab config from environment variables
    meshlabServer = getEnvVar('MESHLAB_SERVER')
    meshlabRoot = getEnvVar('MESHLAB_ROOT')

    if meshlabServer and meshlabRoot:
        scriptFile = os.path.join(meshlabRoot, 'scripts/' + withSystemSuffix(script) + '.mlx')
        subprocess.call([meshlabServer, '-i', inputFile, '-o', outputFile, '-s', scriptFile])
    else:
        print 'No MESHLAB_SERVER or MESHLAB_ROOT found. Check your environment variables.'


def generateTestSTL(meshIds, meshBank, outName):
    '''
    generate the test stl file for checking mesh structure
    '''
    testMesh = TstlMesh()
    for i in meshIds:
        v = meshBank.triDict[i]['vertex']
        for vx in meshBank.triDict[i]['neighbor']:
            tri = [v, vx[0], vx[1]]
            testMesh.appendMeshFace(tri)

    testMesh.appendMeshDone()
    testMesh.meshSave(outName)

def meshBoundaryLoop(tMesh, boundaryV):
    #>>> DiGraph = { node: [connected_nodes] }
    DiG = {}
    boundaryLoop = []
    
    def vxCheck(DiG, v1, v2, v3, vx1, vx2):
        v2_count, v3_count = 0, 0  # counting the appearing of edge
        for _n in tMesh.triDict[v1]['neighbor']:
            for _v in _n:  # each neighbor has x vertex
                if np.array_equal(_v, vx1):
                    v2_count += 1
                if np.array_equal(_v, vx2):
                    v3_count += 1
            
        if tMesh.isEdgeVertex(v1):
            if v1 not in DiG.keys():
                DiG[v1] = []
            if tMesh.isEdgeVertex(v2) and v2_count == 1:
                DiG[v1].append(v2)   # >>> connect to v2
                if v2 not in DiG.keys():
                    DiG[v2] = []
                DiG[v2].append(v1)                    
            if tMesh.isEdgeVertex(v3) and v3_count == 1:
                DiG[v1].append(v3)   # >>> connect to v3
                if v3 not in DiG.keys():
                    DiG[v3] = []
                DiG[v3].append(v1)
    
    
    #>>> constructe Direct graph
    for i in xrange(len(boundaryV)):
        v1 = tMesh.vertexTableLookUp(boundaryV[i][0]) # v1: target vertex
        v2 = tMesh.vertexTableLookUp(boundaryV[i][1])
        v3 = tMesh.vertexTableLookUp(boundaryV[i][2])        
        #print 'v1, v2, v3: ', v1, v2, v3
        
        vxCheck(DiG, v1, v2, v3, boundaryV[i][1], boundaryV[i][2])
        vxCheck(DiG, v2, v1, v3, boundaryV[i][0], boundaryV[i][2])
        vxCheck(DiG, v3, v1, v2, boundaryV[i][0], boundaryV[i][1])
        
    #print 'DiG = ', DiG
    #G = Graph(DiG)
    #boundaryDiG = DiGraph(G)
    #print 'boundaryDiG = ', boundaryDiG
    
    # ==== remove duplicated index
    for i in DiG.keys():
        DiG[i] = list(set(DiG[i]))

    #print 'DiG = ', DiG
    #print 'DiG key = ', DiG.keys()
    #>>> connected graph
    G = Graph(DiG)
    boundaryDiG = DiGraph(G)

    restIdx = sorted(DiG.keys())
    loopCount = 0   # handle the infinit loop(due to non-manifold edge)
    while len(restIdx) > 0:
        selectId = restIdx[0]
        tmpList = []
        for bfs in bfs_edges(boundaryDiG, selectId):
            for i in xrange(2):
                tmpList.append(bfs[0])
                tmpList.append(bfs[1])
        tmpList = sorted(list(set(tmpList)))
        boundaryLoop.append(copy.copy(tmpList))

        restIdx = [item for item in restIdx if item not in set(tmpList)]
        loopCount += 1
        if loopCount > 20:
            break
        #print '*restIdx = ', restIdx

    #print 'boundaryLoop = ', boundaryLoop
    boundaryLoop = sorted([(len(i), i) for i in boundaryLoop])
    #print 'after sort boundaryLoop = ', boundaryLoop

    return boundaryLoop, DiG

class TstlMesh(object):
    def __init__(self, numXBucket=400):
        #self.fmesh = None  # save the mesh from stl file
        # ==== save the mesh that cut from program
        # >>> self.vectors shape = (#row, 3x3) <<<
        # >>> self.normals shape = (#row, 3) <<<
        self.vectors, self.normals = [], []
        self.range = None

        # ==== triangular mesh structure
        self.triDict = {}
        # XDict: bucket vectices into X coord, to look up the vertex and vertex index
        # nXDict: bucket normal into X coord, to eliminate save the duplicated Tri. face
        self.XDict, self.nXDict = {}, {}

        self.meshCentroid = None

        # >>>> initial normal lookup table <<<<
        self.numXBucket = numXBucket #200 #100
        self._normalTable()
    
    @property
    def stlRange(self):
        if self.range is not None:
            return self.range
        else:
            print 'None of range, return None'
            return None

    @property
    def length(self):
        if len(self.vectors) > 0:
            return len(self.vectors)
        else:
            print 'stl return length error'
            return 0

    def getAvgVertex(self, vid):
        if len(self.triDict) <= 0:
            self.setTriStructure()

        vertices = [np.array(self.triDict[vid]['vertex'])]
        for e in self.triDict[vid]['edges']:
            vertices.append(e)

        return np.mean(np.array(vertices), axis=0)
    
    def getAvgNormal(self, vid):
        if len(self.triDict) <= 0:
            self.setTriStructure()
        
        avgNormal = np.array([0.0, 0.0, 0.0])
        vx0 = np.array(self.triDict[vid]['vertex'])
        for vx in self.triDict[vid]['neighbor']:
            v1 = np.array(vx[0])-vx0
            v2 = np.array(vx[1])-vx0
            norm = np.cross(v1, v2)
            unitN = norm/np.linalg.norm(norm)
            unitN = stlRound(unitN)
            avgNormal = avgNormal+unitN
        
        avgNormal=avgNormal/float(len(self.triDict[vid]['vertex']))
        return avgNormal

    def setMeshCentroid(self):
        if len(self.vectors) > 0:
            self.meshCentroid = np.mean(np.mean(self.vectors, axis=1), axis=0)
            self.meshCentroid = stlRound(self.meshCentroid)
        else:
            print 'stl find mesh centroid error'
    
    def moveMesh(self, location):
        if len(self.vectors) > 0:
            self.vectors = self.vectors + location
            tmpRange = self.range.reshape(2, 3) + location
            # >>>> update range <<<<
            self.range = tmpRange.reshape(6)
        else:
            print('ERROR <TstlMesh@moveMesh> no mesh can be moved')

    def moveMesh2Centroid(self):
        if self.meshCentroid is not None:
            if len(self.vectors) > 0:
                tmpCentroid = np.array(self.meshCentroid)
                #tmpCentroid[2] = 0.0
                self.vectors = self.vectors - tmpCentroid
                # >>>> update range
                tmpRange = self.range.reshape(2, 3) - tmpCentroid
                self.range = tmpRange.reshape(6)
            else:
                print('ERROR <TstlMesh@moveMesh2Centroid> no mesh can be moved')
        else:
            print('ERROR <TstlMesh@moveMesh2Centroid> no mesh centroid, move fail')
    
    def shiftMeshCentroid(self, shift):
        if self.meshCentroid is not None:
            if len(self.vectors) > 0:
                shift = np.array(shift)
                self.meshCentroid = self.meshCentroid + np.array(shift)
                self.vectors = self.vectors - shift #+ shift
                # >>>> update range <<<<
                tmpRange = self.range.reshape(2, 3) - shift #+ shift
                self.range = tmpRange.reshape(6)
            else:
                print('ERROR <TstlMesh@shiftMeshCentroid> no mesh can be moved')
        else:
            print('ERROR <TstlMesh@shiftMeshCentroid> no mesh centroid, shift fail')

    def appendMeshFace(self, face):
        v1 = np.array(face[1]) - np.array(face[0])
        v2 = np.array(face[2]) - np.array(face[0])
        norm = np.cross(v1, v2)

        # >>> This face is not valid
        if np.linalg.norm(norm) == 0:
            return False

        if not self._isInNormalTable(face):
            # ==== normalize input vectors
            inputV = []
            for vx in face:
                inputV.append([vx[0], vx[1], vx[2]])

            if isinstance(self.vectors, np.ndarray):
                self.vectors = copy.deepcopy(self.vectors.tolist())
                self.normals = copy.deepcopy(self.normals.tolist())

            self.vectors.append(np.array(inputV))
            self.normals.append(norm)
            return True

    def appendMeshDone(self):
        if len(self.vectors) > 0:
            self.vectors = np.array(self.vectors)
            self.normals = np.array(self.normals)
            self.setRange()

    def getSTLFace(self, vid):
        if len(self.vectors) > 0 and vid < len(self.vectors):
            return self.vectors[vid]
        else:
            print 'stl get vector error'

    def getSTLNormal(self, nid):
        if len(self.normals) > 0 and nid < len(self.normals):
            return self.normals[nid]
        else:
            print 'stl get normal error'

    def projectFaces2Dim(self, dim):
        if len(self.vectors) > 0:
            if len(dim) < 2:
                dimStart, dimEnd = dim[0], dim[0]+1
            else:
                dimStart, dimEnd = dim[0], dim[1]
            return np.reshape(self.vectors[::, ::, dimStart:dimEnd], (len(self.vectors)*3, dimEnd-dimStart))
        else:
            print 'get face data error'
            return []

    def setRange(self):
        if len(self.vectors) > 0:
            minValue, maxValue = np.amin(self.vectors, axis=0), np.amax(self.vectors, axis=0)
            minValue, maxValue = np.amin(minValue, axis=0), np.amax(maxValue, axis=0)
            #print 'minValue, maxValue = ', (minValue, maxValue)
            #print np.amin(minValue, axis=0)

            self.range = np.array( [ minValue[0], minValue[1], minValue[2], \
                                     maxValue[0], maxValue[1], maxValue[2] ] )
        else:
            print 'stl set range error'

    def readFromFile(self, fname):
        try:
            fmesh = mesh.Mesh.from_file(fname)
            if len(self.vectors) > 0:
                self.vectors = []
                self.normals = []
            # ==== check the face area ====
            for i in xrange(len(fmesh.vectors)):
                #farea = self._faceArea(fmesh.vectors[i])
                #if (farea-0.0001) < 0.000001:
                #    print 'face area less than criteria'
                #else:
                #    self.vectors.append(fmesh.vectors[i])
                #    self.normals.append(fmesh.normals[i])
                self.vectors.append(fmesh.vectors[i])
                self.normals.append(fmesh.normals[i])
            
            self.vectors = np.array(self.vectors)
            self.normals = np.array(self.normals)
            self.range = np.array( [ fmesh.min_[0], fmesh.min_[1], fmesh.min_[2], \
                                     fmesh.max_[0], fmesh.max_[1], fmesh.max_[2] ] )
            return True
        except IOError:
            print 'Fail to load stl file in:', fname
            return None

    def copyMesh(self, target):
        if target.vectors is not None:
            self.vectors = np.array(target.vectors)
        if target.normals is not None:
            self.normals = np.array(target.normals)
        self.range = copy.copy(target.stlRange)

    def meshSave(self, namePath):
        if len(self.vectors) > 0:
            _mesh = np.zeros(len(self.vectors), dtype = mesh.Mesh.dtype)
            _mesh['vectors'] = self.vectors
            _mesh['normals'] = self.normals

            meshSave = mesh.Mesh(_mesh, remove_empty_areas=False)
            meshSave.save(namePath)
        else:
            print 'stl save error'

    # >>>> mesh rotate functions <<<<
    def meshZFlip(self):
        if len(self.vectors) > 0:
            R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
            self.vectors = np.dot(self.vectors, R)
            # update normals
            self.normals = np.cross(self.vectors[::,1]-self.vectors[::,0], self.vectors[::,2]-self.vectors[::,0])
            # update range; in this case, we also can do
            # self.range[2]=Zmax*-1, self.range[5]=Zmin*-1
            self.setRange()
            #print self.range
        else:
            print 'stl Z flip error'

    # meshXYRotate
    def meshZAxisRotate(self, rad):
        if len(self.vectors) > 0:
            R = np.array([[math.cos(rad), -1*math.sin(rad), 0], [math.sin(rad), math.cos(rad), 0], [0, 0, 1]])
            self.vectors = np.dot(self.vectors, R)
            # update normals
            self.normals = np.cross(self.vectors[::,1]-self.vectors[::,0], self.vectors[::,2]-self.vectors[::,0])
            # update range
            self.setRange()
            #print 'range = ', self.range
        else:
            print 'stl rotate XY axis error'

    # meshYZRotate
    def meshXAxisRotate(self, rad):
        if len(self.vectors) > 0:
            R = np.array([[1, 0, 0], [0, math.cos(rad), -1*math.sin(rad)], [0, math.sin(rad), math.cos(rad)]])
            self.vectors = np.dot(self.vectors, R)
            # update normals
            self.normals = np.cross(self.vectors[::,1]-self.vectors[::,0], self.vectors[::,2]-self.vectors[::,0])
            # update range
            self.setRange()
            #print 'range = ', self.range
        else:
            print 'stl rotate YZ axis error'

    # meshXZRotate
    def meshYAxisRotate(self, rad):
        if len(self.vectors) > 0:
            R = np.array([[math.cos(rad), 0, math.sin(rad)], [0, 1, 0], [-1*math.sin(rad), 0, math.cos(rad)]])
            self.vectors = np.dot(self.vectors, R)
            # update normals
            self.normals = np.cross(self.vectors[::,1]-self.vectors[::,0], self.vectors[::,2]-self.vectors[::,0])
            # update range
            self.setRange()
            #print 'range = ', self.range
        else:
            print 'stl rotate XZ axis error'

    # >>> mesh cut functions
    def __afterCutVectorNormalUpdate(self, validIdx):
        # >>> update vectors and normal
        self.vectors = self.vectors[validIdx]
        self.normals = self.normals[validIdx]
        self.setRange()
    
    def meshCutBelowZ(self, Zmin):
        if len(self.vectors) > 0:
            meanZ = np.mean(self.vectors[:, :, 2], axis=1)
            validIdx = np.where(meanZ > Zmin)
            self.__afterCutVectorNormalUpdate(validIdx[0])
            # >>> update vectors and normal
            #self.vectors = self.vectors[validIdx[0]]
            #self.normals = self.normals[validIdx[0]]
            #self.setRange()
        else:
            print 'stl cut below Z error'
    
    def meshCutOutofY(self, Yrange):
        if len(self.vectors) > 0:
            meanY = np.mean(self.vectors[:, :, 1], axis=1)
            validIdx = np.where(np.logical_and(meanY > Yrange[0], meanY < Yrange[1]))
            self.__afterCutVectorNormalUpdate(validIdx[0])
        else:
            print 'stl cut between Y range error'

    def meshCutBelowY(self, Ymin):
        if len(self.vectors) > 0:
            meanY = np.mean(self.vectors[:, :, 1], axis=1)
            validIdx = np.where(meanY > Ymin)
            self.__afterCutVectorNormalUpdate(validIdx[0])
        else:
            print 'stl cut below Y error'

    def meshCutOutofX(self, Xrange):
        if len(self.vectors) > 0:
            meanX = np.mean(self.vectors[:, :, 0], axis=1)
            validIdx = np.where(np.logical_and(meanX > Xrange[0], meanX < Xrange[1]))
            self.__afterCutVectorNormalUpdate(validIdx[0])
        else:
            print 'stl cut between X range error'

    def meshInDistance(self, centroidXY, dist):
        if len(self.vectors) > 0:
            meanXY = np.mean(self.vectors[:, :, 0:2], axis=1)
            dists = np.linalg.norm(meanXY-centroidXY, axis=1)
            validIdx = np.where(dists < dist)
            self.__afterCutVectorNormalUpdate(validIdx[0])
        else:
            print 'stl cut between centroid error'

    # >>> mesh property functions
    def clear(self, numXBucket=400):
        self.vectors, self.normals = [], []
        self.range = None

        # ==== triangular mesh structure
        self.triDict = {}
        # XDict: bucket vectices into X coord, to look up the vertex and vertex index
        # nXDict: bucket normal into X coord, to eliminate save the duplicated Tri. face
        self.XDict, self.nXDict = {}, {}

        self.meshCentroid = None

        # >>>> initial normal lookup table <<<<
        self.numXBucket = numXBucket #200 #100
        self._normalTable()

    def vertexTableLookUp(self, vertex):
        xidx = self._XidxDecode(vertex, self.XDict['Xstart'], self.XDict['Xinterval'])
        for _v in self.XDict[xidx]['vertices']:
            if np.array_equal(vertex, _v[1]):
                return int(_v[0])
        return -1

    def isEdgeVertex(self, vid):
        #print 'vid, len(neighbor), len(edges) = ', (vid, len(self.triDict[vid]['neighbor']), \
        #                                            len(self.triDict[vid]['edges']))
        #print 'vertex = ', self.triDict[vid]['vertex']
        #print 'edges = ', self.triDict[vid]['edges']
        #print 'neighbor = ', self.triDict[vid]['neighbor']
        if len(self.triDict[vid]['neighbor']) != len(self.triDict[vid]['edges']):
            return True
        else:
            return False

    def curvatureMesh(self, kvalue=-0.2):#kvalue=-0.005):
        if len(self.triDict) <= 0:
            self.setTriStructure()

        # >>>> reset the normal table, create the new one for extract curvature
        self._normalTable()

        # >>>> vectors, normals may not necessary?
        meshIds = []
        for vid in sorted(self.triDict.keys()):
            tarP = self.triDict[vid]['vertex']
            sumAng = 0.0
            area = 0.0
            for _vx in self.triDict[vid]['neighbor']:
                v1 = _vx[0] - tarP
                v2 = _vx[1] - tarP
                ang = self._angleV1V2(v1, v2)
                area += np.linalg.norm(np.cross(v1, v2))/6.0
                sumAng += ang

            #Kv = round(2*math.pi-sumAng, 3)
            Kv = round((2*math.pi-sumAng)/area, 3)

            if Kv < kvalue:
                #print 'Kv = ', Kv
                for _vx in self.triDict[vid]['neighbor']:
                    tri = [tarP, _vx[0], _vx[1]]

                    if not self._isInNormalTable(tri):
                        meshIds.append(vid)

        meshIds = list(set(meshIds))
        return meshIds

    def boundaryMesh(self):
        if len(self.triDict) <= 0:
            self.setTriStructure()

        # >>>> reset the normal table, create the new one for extract curvature
        self._normalTable()

        meshVectors, meshNormals, meshIds = [], [], []
        for vid in sorted(self.triDict.keys()):
            if len(self.triDict[vid]['neighbor']) != len(self.triDict[vid]['edges']):
                sideEdges = []
                for _vx in self.triDict[vid]['edges']:
                    sideCount, faceId = 0, 0
                    for _fid in xrange(len(self.triDict[vid]['neighbor'])):
                        v1 = self.triDict[vid]['neighbor'][_fid][0]
                        v2 = self.triDict[vid]['neighbor'][_fid][1]
                        if np.array_equal(_vx, v1):
                            sideCount += 1
                            faceId = _fid
                        if np.array_equal(_vx, v2):
                            sideCount += 1
                            faceId = _fid

                    sideEdges.append((sideCount, _vx, faceId))

                for i in sideEdges:
                    if i[0] == 1:
                        faceId = i[2]
                        vx = self.triDict[vid]['vertex']
                        vx1, vx2 = self.triDict[vid]['neighbor'][faceId][0], self.triDict[vid]['neighbor'][faceId][1]
                        tri = [vx, vx1, vx2]

                        v1, v2 = vx1 - vx, vx2 - vx
                        norm = np.cross(v1, v2)
                        if not self._isInNormalTable(tri):
                            meshVectors.append(copy.deepcopy(tri))
                            meshNormals.append(norm)
                            meshIds.append(vid)

        #print 'meshIds = ', meshIds
        return np.array(meshVectors), np.array(meshNormals), meshIds

    # >>> tri mesh data structure: 1-ring structure
    def setTriStructure(self):
        # >>>> construct vector lookup table <<<<
        #if self.range is None:
        #    self.setRange()
        
        # if call setTriStructure, reset the self.triDict
        self.setRange()
        self.triDict = {}

        Xstart = self.range[0]  # min of X
        Xinterval = round(((self.range[3]-self.range[0])/float(self.numXBucket)), 2)

        self.XDict['Xstart'] = Xstart
        self.XDict['Xinterval'] = Xinterval
        for i in xrange(self.numXBucket):
            # >>> the item inside the 'vertices': [(vid, vector), ...]
            # >>> the vector is [x1, y1, z1]
            self.XDict[i] = {'vertices': []}

        if len(self.vectors) > 0:
            start_time = time.time()

            for i in xrange(len(self.vectors)):
                face = self.vectors[i]
                normal = self.normals[i]
                #farea = self._faceArea(face)
                #print 'farea = ', farea
                #if (farea-0.0001) < 0.000001:   #(farea-0.0001) < 0.000001
                #    print 'face area less than criteria'
                #else:
                #    self._setTri(face, normal)
                self._setTri(face, normal)

            duration = time.time() - start_time
            print('construct TriStructure with #vertices=%d spending=%.3fs' \
                                %(len(self.triDict.keys()), duration))
        else:
            print('set Tri structure error, no vectors can construct')
    
    def updateTriStructure(self, face):
        v1 = np.array(face[1]) - np.array(face[0])
        v2 = np.array(face[2]) - np.array(face[0])
        norm = np.cross(v1, v2)
        
        #print('update tri structure, previous #vertices=%d' %len(self.triDict.keys()))
        newV = []
        for vx in face:
            newV.append([vx[0], vx[1], vx[2]])
            
        newV = np.array(newV)
        self._setTri(newV, norm)
        #print('update tri structure, updated #vertices=%d' %len(self.triDict.keys()))

    def _setTri(self, face, normal):
        #Xstart, Xinterval = self.XDict['Xstart'], self.XDict['Xinterval']
        for pp in xrange(3):
            v = face[pp]

            # >>> classify vector by X into Xbucket table
            xidx = self._XidxDecode(v, self.XDict['Xstart'], self.XDict['Xinterval'])

            inDict = False
            # >>> lookup the vector in that X Xbucket
            for _v in self.XDict[xidx]['vertices']:
                if np.array_equal(v, _v[1]):
                    # >>> the duplicated vertex, but with diff. Tri. face
                    inDict = True
                    _idx = _v[0]
                    if pp == 0:
                        self._addTri(_idx, face[1], face[2], v, normal)
                        self._addEdge(_idx, face[1], face[2])
                    if pp == 1:
                        self._addTri(_idx, face[0], face[2], v, normal)
                        self._addEdge(_idx, face[0], face[2])
                    if pp == 2:
                        self._addTri(_idx, face[0], face[1], v, normal)
                        self._addEdge(_idx, face[0], face[1])

                    break

            if not inDict:
                # >>> the new vertex
                vidx = len(self.triDict)
                self.XDict[xidx]['vertices'].append((vidx, v))
                self.triDict[vidx] = {'edges': [], 'neighbor': [], 'vertex': v}
                if pp == 0:
                    self._addTri(vidx, face[1], face[2], v, normal)
                    self.triDict[vidx]['edges'].append(copy.deepcopy(face[1]))
                    self.triDict[vidx]['edges'].append(copy.deepcopy(face[2]))
                if pp == 1:
                    self._addTri(vidx, face[0], face[2], v, normal)
                    self.triDict[vidx]['edges'].append(copy.deepcopy(face[0]))
                    self.triDict[vidx]['edges'].append(copy.deepcopy(face[2]))
                if pp == 2:
                    self._addTri(vidx, face[0], face[1], v, normal)
                    self.triDict[vidx]['edges'].append(copy.deepcopy(face[0]))
                    self.triDict[vidx]['edges'].append(copy.deepcopy(face[1]))


    def _addTri(self, vidx, vect1, vect2, vect, norm):
        v1 = vect1 - vect
        v2 = vect2 - vect
        if np.dot(np.cross(v1, v2), norm) > 0:  # two vectors' normal direction same as norm
            self.triDict[vidx]['neighbor'].append([copy.deepcopy(vect1), copy.deepcopy(vect2)])
        else:
            self.triDict[vidx]['neighbor'].append([copy.deepcopy(vect2), copy.deepcopy(vect1)])

    def _addEdge(self, vidx, v1, v2):
        newV1, newV2 = True, True
        for _v in self.triDict[vidx]['edges']:
            if np.array_equal(_v, v1):
                newV1 = False
            if np.array_equal(_v, v2):
                newV2 = False

        if newV1:
            self.triDict[vidx]['edges'].append(copy.deepcopy(v1))
        if newV2:
            self.triDict[vidx]['edges'].append(copy.deepcopy(v2))

    def _normalTable(self):
        # >>> unit normal vector range = -1.0 ~ 1.0
        Xstart = -1.0
        Xinterval = round(2.0/float(self.numXBucket), 2)  #(max(nX)-min(nX))/100.0

        self.nXDict['Xstart'] = Xstart
        self.nXDict['Xinterval'] = Xinterval
        for i in xrange(self.numXBucket):
            self.nXDict[i] = {'vertices': []}

    def _isInNormalTable(self, v):
        # >>>> v type as array with values=[ [x1, y1, z1], [x2, y2, z2], [x3, y3, z3] ]
        v1 = v[1] - v[0]
        v2 = v[2] - v[0]
        meanV = np.mean(v, axis=0)
        meanV = stlRound(meanV)
        norm = np.cross(v1, v2)
        unitN = norm/np.linalg.norm(norm)
        unitN = stlRound(unitN)

        #>>>> coding to X bucket
        xidx = self._XidxDecode(unitN, self.nXDict['Xstart'], self.nXDict['Xinterval'])

        #>>>> check the items in the bucket
        inDict = False
        for _n in xrange(len(self.nXDict[xidx]['vertices'])):
            if np.array_equal(unitN, self.nXDict[xidx]['vertices'][_n][0]) and \
               np.array_equal(meanV, self.nXDict[xidx]['vertices'][_n][1]):
                   inDict = True

        if not inDict:
            self.nXDict[xidx]['vertices'].append((unitN, meanV))

        return inDict

    def _XidxDecode(self, vertex, Xstart, Xinterval):
        xidx = int((vertex[0]-Xstart)/Xinterval)
        if xidx > self.numXBucket-1:
            xidx = self.numXBucket-1
        if xidx < 0:
            xidx = 0

        return xidx

    def _angleV1V2(self, v1, v2):
        unitV1 = v1/np.linalg.norm(v1)
        unitV2 = v2/np.linalg.norm(v2)
        #return np.arccos(np.clip(np.dot(unitV1, unitV2), -1.0, 1.0))
        return np.arccos(np.clip(np.dot(unitV1, unitV2), -1.0, 1.0))
    
    def _faceArea(self, face):
        v1 = np.array(face[1]) - np.array(face[0])
        v2 = np.array(face[2]) - np.array(face[0])
        area = np.linalg.norm(np.cross(v1, v2))/6.0
        return area
#>>>>>>>>   END  >>>>>>>>>>
