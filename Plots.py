import numpy as np
import os, fnmatch
import matplotlib.pyplot as plt
import matplotlib

def find_files(directory, pattern='watershed_*.npz'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    if len(files) == 0:
    	raise Exception("Could not find any files")
    return np.sort(files)

def sample(files, layer=100, mode='labels'):
	"""mode can be labels, EDT, smEDT, markers"""
	Nplots = 12
	ncols = 2
	if len(files) > Nplots: files = files[:Nplots]
	nrows = len(files)/ncols
	if len(files)%ncols > 0: nrows +=1
	fig, axarr = plt.subplots(nrows, ncols)
	if nrows == 1:
		axarr = axarr[np.newaxis, :]
	if mode in ['labels', 'markers']:
		carr = np.random.rand(256, 3); carr[0,:] = 0
		cmap = matplotlib.colors.ListedColormap(carr)
	else:
		cmap = 'gnuplot'
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
	#DIR = '/data2/lin0_logz10-15_zeta40/Boxes/'
	#DIR = '/home/yunfanz/Data/21cmFast/Boxes/'
	DIR = './NPZ/'
	files = find_files(DIR)
	sample(files, mode='labels')