# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

"""
This example demonstrates how to create a sphere.
"""

import imageio
from vispy import scene
from vispy.visuals.transforms import STTransform
import sys, os, fnmatch
import numpy as np
from skimage import measure, morphology, segmentation
from scipy import ndimage
from vispy import app, scene
from IO_utils import *
file = './NPZ/watershed_z10.0.npz'
DIR = './NPZ/'
TreeDict = load_obj('allnodeTree')
output_filename = 'animation.gif'


def get_border(data, labels=None):
  if labels is None:
    R = measure.regionprops(data)
    r = sorted(R, key=lambda r: r.area, reverse=True)[0]
    labels = [r.label]
  border_images = []
  for label in labels:
    bubblemask = data == label
    eroded_image = ndimage.binary_erosion(bubblemask, np.ones((3,3,3)), border_value=0)
    border = bubblemask ^ eroded_image
    border_images.append(border)
  #import IPython; IPython.embed()
  #X,Y,Z = np.where(border_image)
  return border_images

def get_scenes(file, view, canvas, labels=None):
  print file
  data = np.load(file)['labels']
  vol1 = np.load(file)['EDT']
  data = measure.label(data, connectivity=3)
  border_images = get_border(data, labels=labels)
  
  #print X.shape, Y.shape, Z.shape

  # Create isosurface visual
  #surface = scene.SurfacePlot(x=X, y=Y, z=Z, parent=view.scene)
  Ss = []
  for border in border_images:
    color = (np.random.random(), np.random.random(), np.random.random(), 1)
    #color = (0.5,0.6,0.7,1)
    surface1 = scene.visuals.Isosurface(border, level=1.0,
                                        color=color, shading='smooth',
                                        parent=view.scene)
    Ss.append(surface1)
  volume1 = scene.visuals.Volume(vol1, parent=view.scene, threshold=0.5,
                                 emulate_texture=False)
  t1 = scene.visuals.Text(file, parent=canvas.scene, color='yellow')
  t1.font_size = 16
  t1.pos = canvas.size[0] // 2, canvas.size[1] // 4
  return Ss, volume1, t1

#surface.transform = scene.transforms.STTransform(translate=(-25, -25, -50))
# Create a canvas with a 3D viewport
canvas = scene.SceneCanvas(keys='interactive')
view = canvas.central_widget.add_view()


# t1 = scene.visuals.Text('Text in root scene (24 pt)', parent=canvas.scene, color='red')
# t1.font_size = 16
# t1.pos = canvas.size[0] // 2, canvas.size[1] // 4
# Add a 3D axis to keep us oriented
axis = scene.visuals.XYZAxis(parent=view.scene)
# s = scene.transforms.STTransform(scale=(50, 50, 50, 1))
# affine = s.as_matrix()
# axis.transform = affine
# Use a 3D camera
# Manual bounds; Mesh visual does not provide bounds yet
# Note how you can set bounds before assigning the camera to the viewbox
cam = scene.TurntableCamera(elevation=30, azimuth=0, fov=100.)
cam.set_range((0, 200), (0, 200), (0, 200))
view.camera = cam


files = find_files(DIR)
writer = imageio.get_writer('animation.gif')

for n, file in enumerate(files):
  s, v, t = get_scenes(file, view, canvas, labels=TreeDict[n])
  im = canvas.render()
  writer.append_data(im)
  for surf in s:
    surf.visible = False
  v.visible = False
  t.visible = False

writer.close()
