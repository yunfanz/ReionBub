import numpy as np
import os
import fnmatch
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from watershed import *
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from joblib import Parallel, delayed

#See
#segmentation.mark_boundaries works in 2D

def plot_3d(image, threshold=0.5, n_bubble=1):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    #p = image.transpose(2,1,0)
    R = measure.regionprops(image)
    top_n = sorted([(r.area, r.label) for r in R])[::-1][:n_bubble]
    top_n_R = [R[i-1] for a,i in top_n]


    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(111, projection='3d')
    for r in top_n_R:
        print r.label, r.area
        p = r.image
        verts, faces = measure.marching_cubes(p, threshold)
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        print 'done marching cubes'
        mesh = Poly3DCollection(verts[faces], alpha=0.1)
        face_color = [0.5, 0.5, 1]
        mesh.set_facecolor(face_color)
        #ax.add_collection3d(mesh)

    #ax.set_xlim(0, p.shape[0])
    #ax.set_ylim(0, p.shape[1])
    #ax.set_zlim(0, p.shape[2])

    #plt.show()
    return mesh


def plot_wireframe(image, n_bubble=1):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    #p = image.transpose(2,1,0)
    R = measure.regionprops(image)
    print 'total bubbles', len(R)
    top_n = sorted([(r.area, r.label) for r in R])[::-1][:n_bubble]
    top_n_R = [R[i-1] for a,i in top_n]
    neighborhood = ndimage.morphology.generate_binary_structure(3, 2)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for r in top_n_R:
        print r.label, r.area
        p = r.image
        outline = ndimage.morphological_gradient(p, structure=neighborhood)
        print outline.shape
        ax.plot_wireframe(np.where(outline))

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def find_files(directory, pattern='x_H_nohalos'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files
def plot_distr(path):
    files = 
