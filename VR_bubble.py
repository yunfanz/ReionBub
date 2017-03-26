# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# vispy: gallery 2

"""
Demonstrating a single scene that is shown in four different viewboxes,
each with a different camera.
"""

# todo: the panzoom camera sometimes work, sometimes not. Not sure why.
# we should probably make iterating over children deterministic, so that
# an error like this becomes easier to reproduce ...

import sys
from IO_utils import *
from vispy import app, scene, io
from skimage import measure, morphology, segmentation
from scipy import ndimage
import imageio
canvas = scene.SceneCanvas(keys='interactive')
canvas.size = 1024, 600
canvas.show()
n_steps = 72
step_angle = 2.
step = 2
axis = [0, 0., 1.]
# Create two ViewBoxes, place side-by-side
vb1 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
vb2 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)

scenes = vb1.scene, vb2.scene

# Put viewboxes in a grid
grid = canvas.central_widget.add_grid()
grid.padding = 6
grid.add_widget(vb1, 0, 0)
grid.add_widget(vb2, 0, 1)

# Create some visuals to show
# AK: Ideally, we could just create one visual that is present in all
# scenes, but that results in flicker for the PanZoomCamera, I suspect
# due to errors in transform caching.
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
    #color = (np.random.random(), np.random.random(), np.random.random(), 1)
    color = (0.1,0.6,0.7,0.5)
    surface1 = scene.visuals.Isosurface(border, level=1.0,
                                        color=color, shading='smooth',
                                        parent=view.scene)
    Ss.append(surface1)
  volume1 = scene.visuals.Volume(vol1, parent=view.scene, threshold=0.5,
                                 emulate_texture=False)
  return Ss, volume1

file = './NPZ/watershed_z11.99.npz'
DIR = './NPZ/'
TreeDict = load_obj('allnodeTree')
s1, v1 = get_scenes(file, vb1, canvas, labels=TreeDict[8])
s2, v2 = get_scenes(file, vb2, canvas, labels=TreeDict[8])
#vol1 = np.load(io.load_data_file('volume/stent.npz'))['arr_0']
#volume1 = scene.visuals.Volume(vol1, parent=scenes)
#volume1.transform = scene.STTransform(translate=(0, 0, 10))

# Assign cameras

# vb1.camera = scene.FlyCamera(center=(0,0,0), fov=80)
# vb2.camera = scene.FlyCamera(center=(10,0,0), fov=80)
# vb1.camera.set_range((0, 200), (0, 200), (0, 200))
vb1.camera = scene.TurntableCamera(elevation=0, azimuth=0, fov=120)
vb2.camera = scene.TurntableCamera(elevation=0, azimuth=1, fov=120)
# If True, show a cuboid at each camera
if False:
    cube = scene.visuals.Cube((3, 3, 5))
    cube.transform = scene.STTransform(translate=(0, 0, 6))
    for vb in (vb1, vb2, vb3, vb4):
        vb.camera.parents = scenes
        cube.add_parent(vb.camera)

writer = imageio.get_writer('animation.gif')
for i in range(n_steps):
    im = canvas.render()
    writer.append_data(im)
    vb1.camera.transform.translate((0,step,0))
    vb2.camera.transform.translate((0,step,0))
writer.close()
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()
