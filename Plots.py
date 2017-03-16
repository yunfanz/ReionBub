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
	if len(files) > Nplots: files = files[:Nplots]
	fig, axarr = plt.subplots(3, 4)
	if mode in ['labels', 'markers']:
		carr = np.random.rand(256, 3); carr[0,:] = 0
		cmap = matplotlib.colors.ListedColormap(carr)
	else:
		cmap = 'gnuplot'
	for n in xrange(len(files)):
		file = files[n]
		img = np.load(file)[mode][layer]
		axarr[n/4, n%4].imshow(img, cmap=cmap)
		axarr[n/4, n%4].set_title(file.split('/')[-1])
	plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
	plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
	plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
	plt.setp([a.get_yticklabels() for a in axarr[:, 2]], visible=False)
	plt.setp([a.get_yticklabels() for a in axarr[:, 3]], visible=False)
	plt.show()
	return

if __name__=='__main__':
	DIR = '/data2/lin0_logz10-15_zeta40/Boxes/'
	#DIR = './NPZ/'
	files = find_files(DIR)
	sample(files, mode='markers')