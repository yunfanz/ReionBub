import cosmolopy.perturbation as pb
import cosmolopy.density as cd
import numpy as np
from IO_utils import *
import matplotlib.pyplot as plt
import matplotlib, seaborn as sns
from scipy import ndimage, interpolate
from tocmfastpy import *
from skimage import measure, morphology, segmentation
from Choud14 import *
import pandas as pd



def rescale(image, factor):

    new_real_shape = np.asarray(image.shape) * factor
    new_shape = np.round(new_real_shape)
    real_factor = new_shape / np.asarray(image.shape)
    print real_factor
    new_image = ndimage.interpolation.zoom(image, real_factor, mode='nearest')
    return new_image, real_factor

def find_deltax(directory, z):
	pattern = 'updated_smoothed_deltax_z0{}'.format(z)+'*'
	print pattern
	return find_files(directory, pattern=pattern)

if __name__=="__main__":
	#DIR = '/home/yunfanz/Data/21cmFast/Boxes/'
	DIR = '/data2/21cmFast/Barrierz_12/Boxes/'
	# z = 11.99
	# npzfile = './NPZ/dwatershed_z{}.npz'.format(z)
	# labels = np.load(npzfile)['labels']
	# labels = measure.label(labels, connectivity=3)
	# deltax_file = find_deltax(DIR, z)[0]
	# deltax_image = boxio.readbox(deltax_file).box_data
	# print labels.shape, deltax_image.shape
	# #deltax_image = rescale(deltax_image, 0.5)
	# R = measure.regionprops(labels, intensity_image=deltax_image)
	# R = R[:1000]
	# RE = [r.equivalent_diameter/2 for r in R]
	# ES = ESets(z=z)
	# R0L = ES.R0(RE)
	# deltax = np.asarray([r.mean_intensity for r in R])
	# deltax /= ES.fgrowth

	# S = [sig0(rl) for rl in R0L]
	# S1, deltax1 = S, deltax


	z = 12.00
	#npzfile = './NPZ/dwatershed_z{}.npz'.format(z)
	wspattern = 'dwatershed_z{}*.npz'.format(z)
	npzfiles = find_files('./NPZ/', pattern=wspattern)
	deltax_files = find_deltax(DIR, z)
	dframes = []
	for i, npzfile in enumerate(npzfiles):
		labels = np.load(npzfile)['labels']
		scale = np.load(npzfile)['scale']
		print scale
		labels = measure.label(labels, connectivity=3)
		deltax_file = deltax_files[i]
		print deltax_file
		b1 = boxio.readbox(deltax_file)
		deltax_image = b1.box_data
		print labels.shape, deltax_image.shape
		#deltax_image = rescale(deltax_image, 0.5)
		R = measure.regionprops(labels, intensity_image=deltax_image)
		print len(R)
		R = R[:5000]
		RE = np.asarray([r.equivalent_diameter/2 for r in R])/scale
		ES = ESets(z=z)
		R0L = ES.R0(RE)
		deltax = np.asarray([r.mean_intensity for r in R])
		deltax /= ES.fgrowth
		S = sig0(R0L)
		dx = 1/scale
		L = b1.param_dict['BoxSize']
		df = pd.DataFrame({'RL': R0L, 'RE':RE, 'S': S, 'deltax':deltax, 'BoxSize': L})
		df = df.loc[df['RE'] > 10*dx]
		df = df.loc[df['RE'] < L/10]
		dframes.append(df)
	df = pd.concat(dframes)
	#import IPython; IPython.embed()
	plt.figure()
	
	sns.regplot('S','deltax', df, scatter_kws={'hue': "BoxSize"})
	plt.show()






