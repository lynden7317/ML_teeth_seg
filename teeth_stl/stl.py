# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 09:54:18 2018

@author: lynden
"""
import copy
import math
import numpy as np
from sys import version_info

#from .. import util
from teeth_stl import util
from teeth_stl import transformations
from teeth_stl import triangles
from teeth_stl import mesh

#print util.is_sequence([1, 2, 3])

class HeaderError(Exception):
    # the exception raised if an STL file object doesn't match its header
    pass

# define a numpy datatype for the data section of a binary STL file
_stl_dtype = np.dtype([('normals', np.float32, (3)),
                       ('vertices', np.float32, (3, 3)),
                       ('attributes', np.uint16)])

# define a numpy datatype for the header of a binary STL file
_stl_dtype_header = np.dtype([('header', np.void, 80),
                              ('face_count', np.int32)])

# a flag to check for Python 3
PY3 = version_info.major >= 3


def write_encoded(file_obj,
                  stuff,
                  encoding='utf-8'):
    
    binary_file = 'b' in file_obj.mode
    string_stuff = isinstance(stuff, str)#basestring)
    binary_stuff = isinstance(stuff, bytes)
    
    if not PY3:
        file_obj.write(stuff)
    elif binary_file and string_stuff:
        file_obj.write(stuff.encode(encoding))
    elif not binary_file and binary_stuff:
        file_obj.write(stuff.decode(encoding))
    else:
        file_obj.write(stuff)
    file_obj.flush()


def load_stl(file_obj):
    """
    Load an STL file
    
    Returns
    ----------
    loaded: kwargs for a Trimesh constructor with keys:
              vertices:     (n,3) float, vertices
              faces:        (m,3) int, indexes of vertices
              face_normals: (m,3) float, normal vector of each face
    """
    # save start of file obj
    file_pos = file_obj.tell()
    try:
        # check the file for a header which matches the file length
        # if that is true, it is almost certainly a binary STL file
        # if the header doesn't match the file length a HeaderError will be
        # raised
        return load_stl_binary(file_obj)
    except HeaderError:
        # move the file back to where it was initially
        file_obj.seek(file_pos)

def load_stl_binary(file_obj):
    # the header is always 84 bytes long, we just reference the dtype.itemsize
    # to be explicit about where that magical number comes from
    header_length = _stl_dtype_header.itemsize
    header_data = file_obj.read(header_length)
    if len(header_data) < header_length:
        raise HeaderError('Binary STL shorter than a fixed header!')
    
    try:
        header = np.frombuffer(header_data,
                               dtype=_stl_dtype_header)
    except BaseException:
        raise HeaderError('Binary header incorrect type')
    
    try:
        # save the header block as a string
        # there could be any garbage in there so wrap in try
        metadata = {'header':
                    bytes(header['header'][0]).decode('utf-8').strip()}
    except BaseException:
        metadata = {}
    
    # now we check the length from the header versus the length of the file
    # data_start should always be position 84, but hard coding that felt ugly
    data_start = file_obj.tell()
    # this seeks to the end of the file
    # position 0, relative to the end of the file 'whence=2'
    file_obj.seek(0, 2)
    # we save the location of the end of the file and seek back to where we
    # started from
    data_end = file_obj.tell()
    file_obj.seek(data_start)
    
    # the binary format has a rigidly defined structure, and if the length
    # of the file doesn't match the header, the loaded version is almost
    # certainly going to be garbage.
    len_data = data_end - data_start
    len_expected = header['face_count'] * _stl_dtype.itemsize
    
    # this check is to see if this really is a binary STL file.
    # if we don't do this and try to load a file that isn't structured properly
    # we will be producing garbage or crashing hard
    # so it's much better to raise an exception here.
    if len_data != len_expected:
        raise HeaderError('Binary STL has incorrect length in header!')
    blob = np.frombuffer(file_obj.read(), dtype=_stl_dtype)
    
    # all of our vertices will be loaded in order
    # so faces are just sequential indices reshaped.
    faces = np.arange(header['face_count'] * 3).reshape((-1, 3))
    
    result = {'vertices': blob['vertices'].reshape((-1, 3)),
              'face_normals': blob['normals'].reshape((-1, 3)),
              'faces': faces,
              'metadata': metadata}
    
    return result

def export_stl(mesh):
    """
    Convert a Mesh object into a binary STL file.
    Parameters
    ---------
    mesh: Mesh object
    Returns
    ---------
    export: bytes, representing mesh in binary STL form
    """
    header = np.zeros(1, dtype=_stl_dtype_header)
    header['face_count'] = len(mesh.faces)
    
    packed = np.zeros(len(mesh.faces), dtype=_stl_dtype)
    packed['normals'] = mesh.face_normals
    packed['vertices'] = mesh.triangles
    
    export = header.tostring()
    export += packed.tostring()

    return export

def export_mesh(mesh, file_obj):
    was_opened = False
    
    if isinstance(file_obj, str): #basestring):
        was_opened = True
        file_obj = open(file_obj, 'wb')
    
    export = export_stl(mesh)
    if hasattr(file_obj, 'write'):
        result = write_encoded(file_obj, export)
    
    if was_opened:
        file_obj.close()
        
    return result

'''
def bounds_tree(triangles):
    """
    Given a list of triangles, create an r-tree for broad- phase
    collision detection
    Parameters
    ---------
    triangles: (n, 3, 3) list of vertices
    Returns
    ---------
    tree: Rtree object
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError('Triangles must be (n,3,3)!')

    # the (n,6) interleaved bounding box for every triangle
    # (xmin, ymin, zmin, xmax, ymax, zmax)
    triangle_bounds = np.column_stack((triangles.min(axis=1),
                                       triangles.max(axis=1)))
    #print 'tri_bounds = ', triangle_bounds
    tree = util.bounds_tree(triangle_bounds)
    return tree
'''

'''
file_path = 'Step-6-Att-Mandible.stl'
file_obj = open(file_path, 'rb')

stlmesh = load_stl(file_obj)

print stlmesh.keys()
#print stlmesh['metadata']
print len(stlmesh['vertices'])
print stlmesh['faces']

loadMesh = mesh.aaaMesh(vertices=mesh['vertices'], faces=mesh['faces'], face_normals=mesh['face_normals'])

print 'mesh bounding box = ', loadMesh.bounds
print 'centroid = ', loadMesh.centroid

tMatrix = transformations.translation_matrix([10, 0, 0])
#rMatrix = transformations.rotation_matrix(math.pi/2, [0, 0, 1])
loadMesh.apply_transform(tMatrix)

export_mesh(loadMesh, 'test.stl')
'''

'''
tree = loadMesh.triangles_tree()
print 'build the tree done'

intersection = list(tree.intersection((-32, 3, -3, 0, 8, 20)))
#print intersection


newMesh = {'vertices': copy.deepcopy(mesh['vertices']), 'faces': [], 'face_normals': []}

for i in intersection:
    newMesh['faces'].append(loadMesh.faces[i])
    newMesh['face_normals'].append(loadMesh.face_normals[i])

nMesh = aaaMesh(vertices=newMesh['vertices'], faces=newMesh['faces'], face_normals=newMesh['face_normals'])
export_mesh(nMesh, 'test.stl')
'''