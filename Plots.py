import numpy as np
import os, fnmatch
import matplotlib.pyplot as plt
import matplotlib
from IO_utils import *

def sample(files, layer=100, mode='labels', ncols=3, nrows=0):
	"""mode can be labels, EDT, smEDT, markers"""
	if nrows == 0:
		Nplots = len(files)
		nrows = len(files)/ncols
	else:
		Nplots = ncols * nrows
		if len(files) > Nplots: files = files[:Nplots]
		nrows = len(files)/ncols
	if mode in ['labels', 'markers']:
		carr = np.random.rand(256, 3); carr[0,:] = 0
		cmap = matplotlib.colors.ListedColormap(carr)
	else:
		cmap = 'gnuplot'

	if Nplots == 1:
		file = files[0]
		img = np.load(file)[mode][:,layer]
		plt.imshow(img, cmap=cmap)
		plt.title(file.split('/')[-1])
		plt.show
		return
	else:

		if len(files)%ncols > 0: nrows +=1
		fig, axarr = plt.subplots(nrows, ncols)
		if nrows == 1:
			axarr = axarr[np.newaxis, :]

		for n in xrange(len(files)):
			file = files[n]
			img = np.load(file)[mode][:,layer]
			axarr[n/ncols, n%ncols].imshow(img, cmap=cmap)
			axarr[n/ncols, n%ncols].set_title(file.split('/')[-1])
		for i in xrange(nrows-1):
			plt.setp([a.get_xticklabels() for a in axarr[i, :]], visible=False)
		for j in xrange(1,ncols):
			plt.setp([a.get_yticklabels() for a in axarr[:, j]], visible=False)
		plt.show()
		return

if __name__=='__main__':
	o = optparse.OptionParser()
	o.add_option('-d','--dir', dest='DIR', default='./NPZ/')
	o.add_option('-c','--cols', dest='NCOLS', default=3) 
	o.add_option('-r','--rows', dest='NROWS', default=1)
	(opts, args) = o.parse_args()
	files = find_files(opts.DIR, pattern='dwatershed_z12*Iter0*')
	sample(files, mode='labels', ncols=opts.NCOLS)
