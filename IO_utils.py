import pickle
import os, fnmatch, re
import numpy as np
import optparse

def find_files(directory, pattern='watershed_*.npz'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    if len(files) == 0:
    	raise Exception("Could not find any files")
    return np.sort(files)

def find_deltax(directory, z):
    pattern = 'updated_smoothed_deltax_z0{}'.format(z)+'*'
    return find_files(directory, pattern=pattern)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
