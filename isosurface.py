# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

"""
This example demonstrates the use of the Isosurface visual.
"""

import sys, os, fnmatch
import numpy as np
from skimage import measure, morphology, segmentation
from scipy import ndimage
from vispy import app, scene
from IO_utils import *
file = './NPZ/watershed_z10.0.npz'
DIR = './NPZ/'
TreeDict = load_obj('3nodeTree')


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
    surface1.visible = False
    Ss.append(surface1)
  volume1 = scene.visuals.Volume(vol1, parent=view.scene, threshold=0.5,
                                 emulate_texture=False)
  volume1.visible = False
  t1 = scene.visuals.Text(file, parent=canvas.scene, color='yellow')
  t1.font_size = 16
  t1.pos = canvas.size[0] // 2, canvas.size[1] // 4
  t1.visible = False
  return Ss, volume1, t1

#surface.transform = scene.transforms.STTransform(translate=(-25, -25, -50))
# Create a canvas with a 3D viewport
canvas = scene.SceneCanvas(keys='interactive')
view = canvas.central_widget.add_view()


# t1 = scene.visuals.Text('Text in root scene (24 pt)', parent=canvas.scene, color='red')
# t1.font_size = 16
# t1.pos = canvas.size[0] // 2, canvas.size[1] // 4


Surfaces = []; Volumes = []; Texts = []
files = find_files(DIR)
for n, file in enumerate(files):
  s, v, t = get_scenes(file, view, canvas, labels=TreeDict[n])
  Surfaces.append(s)
  Volumes.append(v)
  Texts.append(t)
Surfaces[0][0].visible = True
Volumes[0].visible = True
Texts[0].visible = True
# Add a 3D axis to keep us oriented
axis = scene.visuals.XYZAxis(parent=view.scene)
s = scene.transforms.STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
affine = s.as_matrix()
axis.transform = affine
# Use a 3D camera
# Manual bounds; Mesh visual does not provide bounds yet
# Note how you can set bounds before assigning the camera to the viewbox
cam = scene.TurntableCamera(elevation=30, azimuth=30, fov=60.)
#cam.set_range((0, 200), (-200, 200), (-200, 200))
view.camera = cam

# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
  n = int(event.text) - 1
  if n >= len(Surfaces):
    print "n too large"
    return
  for Ss in Surfaces: 
    for s in Ss:
      s.visible = False
  for v in Volumes: v.visible = False
  for t in Texts: t.visible = False
  for s in Surfaces[n]:
    s.visible = True
  Volumes[n].visible = True
  Texts[n].visible = True
  # elif event.text == '2':
  #   volume1.visible = not volume1.visible
  #   volume2.visible = not volume1.visible
if __name__ == '__main__':

    canvas.show()
    if sys.flags.interactive == 0:
        app.run()
