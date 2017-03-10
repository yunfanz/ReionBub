# Modified from:  vispy: gallery 2
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

"""

Controls:

* 1  - toggle camera between first person (fly), regular 3D (turntable) and
       arcball
* 2  - toggle between volume rendering methods
* 3  - toggle between rendered images
* 4  - toggle between colormaps
* 0  - reset cameras
* [] - decrease/increase isosurface threshold

With fly camera:

* WASD or arrow keys - move around
* SPACE - brake
* FC - move up-down
* IJKL or mouse - look around
"""

from itertools import cycle
from tocmfastpy import *
import numpy as np
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max
import os
import watershed
from vispy import app, scene, io
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.transforms import STTransform

def get_data():

    if not os.path.exists('./basins.npy'):
        PATH = '/home/yunfanz/Data/21cmFast/Boxes/xH_nohalos_z010.00_nf0.865885_eff20.0_effPLindex0.0_HIIfilter1_Mmin4.3e+08_RHIImax20_500_500Mpc'
        d1 = 1 - boxio.readbox(PATH).box_data
        ionized = d1 > 0.999
        ionized = ionized*morphology.remove_small_objects(ionized, 3)  #speeds up later process
        EDT = ndimage.distance_transform_edt(ionized)
        smoothed_arr = np.load('smoothed.npy')
        maxima = watershed.local_maxima(smoothed_arr, ionized, h_transform=False)
        basins = np.where(maxima*EDT>5)
        basins = np.asarray(basins).T
        #import IPython; IPython.embed()
        #basins = morphology.remove_small_objects(maxima, 9)  #speeds up later process
        #markers = measure.label(maxima, connectivity=2)
        #R = measure.regionprops(markers)
        #basins = np.vstack([np.mean(r.coords, axis=0) for r in R])
        np.save('basins.npy', basins)
        np.save('EDT.npy', EDT)
    else:
        EDT = np.load('./EDT.npy')
        basins = np.load('./basins.npy')
    return EDT, basins
vol1, basins = get_data()
#import IPython; IPython.embed()
# Read volume
#vol1 = np.load(io.load_data_file('volume/stent.npz'))['arr_0']
# vol2 = np.load(io.load_data_file('brain/mri.npz'))['data']
# vol2 = np.flipud(np.rollaxis(vol2, 1))

vol2 = np.load('./smoothed.npy')

# Prepare canvas
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
canvas.measure_fps()

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

# Set whether we are emulating a 3D texture
emulate_texture = False

# Create the volume visuals, only one is visible
volume1 = scene.visuals.Volume(vol1, parent=view.scene, threshold=0.5,
                               emulate_texture=emulate_texture)
#volume1.transform = scene.STTransform(translate=(64, 64, 0))
volume2 = scene.visuals.Volume(vol2, parent=view.scene, threshold=0.5,
                               emulate_texture=emulate_texture)
volume2.visible = False

scatter = scene.visuals.Markers()
scatter.set_data(basins, edge_color=None, face_color=(1, 0, 0, 1), size=5)
view.add(scatter)


# Create three cameras (Fly, Turntable and Arcball)
fov = 60.
cam1 = scene.cameras.FlyCamera(parent=view.scene, fov=fov, name='Fly')
cam2 = scene.cameras.TurntableCamera(parent=view.scene, fov=fov,
                                     name='Turntable')
cam3 = scene.cameras.ArcballCamera(parent=view.scene, fov=fov, name='Arcball')
view.camera = cam2  # Select turntable at first

# Create an XYZAxis visual
axis = scene.visuals.XYZAxis(parent=view)
s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
affine = s.as_matrix()
axis.transform = affine


# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """

# Setup colormap iterators
opaque_cmaps = cycle(get_colormaps())
translucent_cmaps = cycle([TransFire(), TransGrays()])
opaque_cmap = next(opaque_cmaps)
translucent_cmap = next(translucent_cmaps)


# Implement axis connection with cam2
@canvas.events.mouse_move.connect
def on_mouse_move(event):
    if event.button == 1 and event.is_dragging:
        axis.transform.reset()

        axis.transform.rotate(cam2.roll, (0, 0, 1))
        axis.transform.rotate(cam2.elevation, (1, 0, 0))
        axis.transform.rotate(cam2.azimuth, (0, 1, 0))

        axis.transform.scale((50, 50, 0.001))
        axis.transform.translate((50., 50.))
        axis.update()


# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
    global opaque_cmap, translucent_cmap
    if event.text == '1':
        cam_toggle = {cam1: cam2, cam2: cam3, cam3: cam1}
        view.camera = cam_toggle.get(view.camera, cam2)
        print(view.camera.name + ' camera')
        if view.camera is cam2:
            axis.visible = True
        else:
            axis.visible = False
    elif event.text == '2':
        methods = ['mip', 'translucent', 'iso', 'additive']
        method = methods[(methods.index(volume1.method) + 1) % 4]
        print("Volume render method: %s" % method)
        cmap = opaque_cmap if method in ['mip', 'iso'] else translucent_cmap
        volume1.method = method
        volume1.cmap = cmap
        volume2.method = method
        volume2.cmap = cmap
    elif event.text == '3':
        volume1.visible = not volume1.visible
        volume2.visible = not volume1.visible
    elif event.text == '4':
        if volume1.method in ['mip', 'iso']:
            cmap = opaque_cmap = next(opaque_cmaps)
        else:
            cmap = translucent_cmap = next(translucent_cmaps)
        volume1.cmap = cmap
        volume2.cmap = cmap
    elif event.text == '0':
        cam1.set_range()
        cam3.set_range()
    elif event.text != '' and event.text in '[]':
        s = -0.025 if event.text == '[' else 0.025
        volume1.threshold += s
        volume2.threshold += s
        th = volume1.threshold if volume1.visible else volume2.threshold
        print("Isosurface threshold: %0.3f" % th)

# for testing performance
# @canvas.connect
# def on_draw(ev):
# canvas.update()

if __name__ == '__main__':
    print(__doc__)
    app.run()
