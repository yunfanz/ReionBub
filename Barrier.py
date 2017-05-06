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
import optparse


def rescale(image, factor):

    new_real_shape = np.asarray(image.shape) * factor
    new_shape = np.round(new_real_shape)
    real_factor = new_shape / np.asarray(image.shape)
    print real_factor
    new_image = ndimage.interpolation.zoom(image, real_factor, mode='nearest')
    return new_image, real_factor

if __name__=="__main__":
	o = optparse.OptionParser()
	o.add_option('-d','--dir', dest='DIR', default='/home/yunfanz/Data/21cmFast/Boxes/')
	o.add_option('-z','--npz', dest='NPZDIR', default='./NPZ/')
	o.add_option('-l','--lin', dest='LIN', action="store_true")
	(opts, args) = o.parse_args()

	z = 12.00
	#npzfile = './NPZ/dwatershed_z{}.npz'.format(z)
	wspattern = 'dwatershed_z{}_L*.npz'.format(z)
	npzfiles = find_files(opts.NPZDIR, pattern=wspattern)
	#deltax_files = find_deltax(opts.DIR, z)
	dframes = []
	for i, npzfile in enumerate(npzfiles):
		npz_params = boxio.parse_filename(npzfile)
		if npz_params['BoxSize'] != 512:
			continue
		data = np.load(npzfile)
		labels = data['labels']
		scale = data['scale']
		labels = measure.label(labels, connectivity=3)

		#looking for matching deltax file
		#for file in deltax_files:
		#	if (boxio.parse_filename(file)['BoxSize'] == npz_params['BoxSize']) and (boxio.parse_filename(file)['Iteration'] == npz_params['Iteration']):
		#		deltax_file = file
		#		print "Found matching files:", npzfile, deltax_file
		#		break
		deltax_file = args[0]
		
		try:
			deltax_image = np.load(deltax_file)
			p_dict = boxio.parse_filename(deltax_file)
		except:
			b1 = boxio.readbox(deltax_file)
			deltax_image = b1.box_data
			p_dict = b1.param_dict
		#deltax_image = rescale(deltax_image, 0.5)
		R = measure.regionprops(labels, intensity_image=deltax_image)
		print len(R)
		if len(R)> 20000:
			R = R[:20000]
		
		ES = ESets(z=z)
		RE = np.asarray([r.equivalent_diameter/2 for r in R])/scale
		
		#R0L = RE
		deltax = np.asarray([r.mean_intensity for r in R])
		if not opts.LIN:
			deltax /= ES.fgrowth
		
		dx = 1/scale
		L = p_dict['BoxSize']
		df = pd.DataFrame({'RE':RE, 'deltax':deltax, 'BoxSize': L})
		#import IPython; IPython.embed()
		#df = df.loc[df['RE'] > max(5*dx, ES.R0min/2)] # 2 is arbitrary, we really want to compare R0L as below
		#df = df.loc[df['RE'] < L/10]
		if len(df.index) == 0: continue
		try:
			df['R0L'] = ES.R0(df['RE'])
		except(ValueError):
			print 'Below interpolation range, use  RE'
			df['R0L'] = df['RE']
		df = df.loc[df['R0L'] > ES.R0min]
		df['S'] = sig0(df['R0L'])
		dframes.append(df)
	df = pd.concat(dframes)
	SL = np.linspace(0, 1.1*np.amax(df['S']), 100)
	bfzh = BFZH(SL,ES.deltac,ES.smin,ES.K)

	#import IPython; IPython.embed()
	sns.jointplot(x='S', y="deltax", data=df, kind="reg")
	plt.plot(SL, bfzh, 'r', linewidth=3)
	#sns.regplot('S','deltax', df)#, scatter_kws={'hue': "BoxSize"})
	
	plt.show()






