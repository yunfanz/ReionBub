import numpy as np
import os, fnmatch
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage
from skimage import measure, morphology, segmentation

def find_files(directory, pattern='watershed_*.npz'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    if len(files) == 0:
    	raise Exception("Could not find any files")
    return np.sort(files)

def sample(files, connectivity=2, mode='labels'):
	"""mode can be labels, EDT, smEDT, markers"""

	for n in xrange(len(files)):
		file = files[n]
		img = np.load(file)[mode]
		labeled = measure.label(img, connectivity=connectivity)
		R = measure.regionprops(labeled)

	for i in xrange(nrows-1):
		plt.setp([a.get_xticklabels() for a in axarr[i, :]], visible=False)
	for j in xrange(1,4):
		plt.setp([a.get_yticklabels() for a in axarr[:, j]], visible=False)
	plt.show()
	return

if __name__=='__main__':
	#DIR = '/data2/lin0_logz10-15_zeta40/Boxes/'
	DIR = '/home/yunfanz/Data/21cmFast/Boxes/'
	files = find_files(DIR)
	sample(files, mode='labels')