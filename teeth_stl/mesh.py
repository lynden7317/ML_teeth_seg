# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 17:04:01 2018

@author: lynden
"""
import numpy as np

from teeth_stl import transformations
from teeth_stl import triangles
from teeth_stl import util

class aaaMesh(object):
    def __init__(self, 
                 vertices=None,
                 faces=None,
                 face_normals=None,
                 metadata=None):
        
        self._data = {}
        
        if faces is not None:
            self.faces = faces
        if vertices is not None:
            self.vertices = vertices
        if face_normals is not None:
            self.face_normals = face_normals
    
    @property
    def vertices(self):
        return self._data['vertices']
    
    @vertices.setter
    def vertices(self, values):
        if values is not None:
            pass
        self._data['vertices'] = values
    
    @property
    def faces(self):
        return self._data['faces']
    
    @faces.setter
    def faces(self, values):
        if values is None:
            values = []
        values = np.asanyarray(values, dtype=np.int64)
        self._data['faces'] = values
    
    @property
    def triangles(self):
        """
        Returns
        ---------
        triangles : (n, 3, 3) float
          Points of triangle vertices
        """
        triangles = self.vertices.view(np.ndarray)[self.faces]
        #print 'tri = ', triangles
        return triangles
    
    @property
    def triangles_center(self):
        """
        The center of each triangle
        Returns
        ---------
        triangles_center : (len(self.faces), 3) float
          Center of each triangular face
        """
        triangles_center = self.triangles.mean(axis=1)
        return triangles_center
    
    @property
    def area_faces(self):
        """
        The area of each face in the mesh.
        Returns
        ---------
        area_faces : (n,) float
          Area of each face
        """
        area_faces = triangles.area(crosses=self.triangles_cross,
                                    sum=False)
        return area_faces
    
    @property
    def triangles_cross(self):
        """
        The cross product of two edges of each triangle.
        Returns
        ---------
        crosses : (n, 3) float
          Cross product of each triangle
        """
        crosses = triangles.cross(self.triangles)
        return crosses
    
    @property
    def centroid(self):
        """
        The point in space which is the average of the triangle centroids
        weighted by the area of each triangle.
        This will be valid even for non- watertight meshes,
        unlike self.center_mass
        Returns
        ----------
        centroid : (3,) float
          The average vertex weighted by face area
        """

        # use the centroid of each triangle weighted by
        # the area of the triangle to find the overall centroid
        
        centroid = np.average(self.triangles_center,
                              axis=0,
                              weights=self.area_faces)
        
        #centroid = self.bounds.mean(axis=0)
        
        return centroid
    
    @property
    def face_normals(self):
        return self._data['face_normals']
    
    @face_normals.setter
    def face_normals(self, values):
        if values is not None:
            pass
        self._data['face_normals'] = values
    
    @property
    def bounds(self):
        """
        The axis aligned bounds of the mesh.
        Returns
        -----------
        bounds : (2, 3) float
          Bounding box with [min, max] coordinates
        """
        in_mesh = self.triangles.reshape((-1, 3))
        bounds = np.vstack((in_mesh.min(axis=0),
                            in_mesh.max(axis=0)))
        return bounds
    
    def apply_transform(self, matrix):
        """
        Transform mesh by a homogenous transformation matrix.
        
        Parameters
        ----------
        matrix : (4, 4) float
          Homogenous transformation matrix
        """
        # get c-order float64 row-major matrix (C-style)
        matrix = np.asanyarray(matrix,
                               order='C',
                               dtype=np.float64)
        
        # only support homogenous transformations
        if matrix.shape != (4, 4):
            raise ValueError('Transformation matrix must be (4,4)!')
        
        # new vertex positions
        new_vertices = transformations.transform_points(
            self.vertices,
            matrix=matrix)
        
        new_face_normals = util.unitize(
                transformations.transform_points(
                    self.face_normals,
                    matrix=matrix,
                    translate=False))
        
        # assign the new values
        self.vertices = new_vertices
        self.face_normals = new_face_normals
    
    def apply_flip(self):
        matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        tMatrix = np.array([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        # new vertex positions
        new_vertices = transformations.transform_points(
            self.vertices,
            matrix=matrix)
        
        new_faces = transformations.transform_points(
            self.faces,
            matrix=tMatrix)
        
        new_face_normals = util.unitize(
                transformations.transform_points(
                    self.face_normals,
                    matrix=matrix,
                    translate=False))

        # assign the new values
        self.vertices = new_vertices
        self.face_normals = new_face_normals
        self.faces = new_faces
        
    
    
    '''
    def triangles_tree(self):
        """
        An R-tree containing each face of the mesh.
        Returns
        ----------
        tree : rtree.index
          Each triangle in self.faces has a rectangular cell
        """
        tree = bounds_tree(self.triangles)
        return tree
    '''

