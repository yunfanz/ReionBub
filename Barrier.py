import cosmolopy.perturbation as pb
import cosmolopy.density as cd
import numpy as np
from IO_utils import *
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage
from tocmfastpy import *
from skimage import measure, morphology, segmentation


dsig0 = np.load('Choud14/sig0.npz')
sig0Rl,sig0arr = dsig0['arr_0'],dsig0['arr_1']
fsig0 = interp1d(sig0Rl,sig0arr)
def sig0(RL):
	return fsig0(RL)


def rescale(image, factor):

    new_real_shape = image.shape * factor
    new_shape = np.round(new_real_shape)
    real_factor = new_shape / image.shape
    
    new_image = ndimage.interpolation.zoom(image, real_factor, mode='nearest')
    
    return new_image, real_factor
def find_deltax(directory, z):
	pattern = 'updated_smoothed_deltax_z0{}'.format(z)+'*'
	return find_files(directory, pattern=pattern)

if __name__=="__main__":
	DIR = '~/Data/21cmFast/Boxes/'
	z = 11.99
	npzfile = './NPZ/watershed_z{}.npz'.format(z)
	labels = np.load(npzfile)['labels']
	labels = measure.label(labels, connectivity=3)
	deltax_image = find_deltax(DIR, z)[0]
	deltax_image = rescale(deltax, 0.5)
	R = measure.regionprops(labels, intensity_image=deltax_image)
	deltax = [r.mean_intensity for r in R]
	RE = [r.equivalent_diameter/2 for r in R]
	RL = np.array(RE)
	#[TODO] need to convert to lagrangian coordinates
	S = [sig0(rl) for rl in RL]
	plt.figure()
	plt.scatter(S, deltax)
	plt.show()






