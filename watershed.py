import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max
from tocmfastpy import *

def h_transform(arr, h, neighborhood):
    diff = ndimage.filters.maximum_filter(arr, footprint=neighborhood) - arr

def local_maxima(arr, ionized, threshold_h=0., smooth=2):

    neighborhood = ndimage.morphology.generate_binary_structure(len(arr.shape),smooth)
    if False:
        local_max = (ndimage.filters.maximum_filter(arr, footprint=neighborhood)==arr)
        background = (arr==0)
        eroded_background = ndimage.morphology.binary_erosion(
            background, structure=neighborhood, border_value=1)
        maxima = local_max ^ eroded_background
    else:
        maxima = peak_local_max(arr, labels=ionized, footprint=neighborhood, indices=False)
    return maxima #np.where(detected_maxima)


def watershed_3d(image):
    #Creation of the internal Marker
    ionized = image > 0.
    EDT = ndimage.distance_transform_edt(ionized)
    #marker_internal = apply_func_3d(segmentation.clear_border, marker_internal) #this removes bubbles on the sides
    maxima = local_maxima(EDT, ionized)
    markers = ndimage.label(maxima)[0]
    labels = morphology.watershed(-EDT, markers, mask=ionized)
    
    return labels, markers, EDT


if __name__ == '__main__':
    b1 = boxio.readbox('../pkgs/21cmFAST/TrialBoxes/xH_nohalos_z008.06_nf0.604669_eff20.0_effPLindex0.0_HIIfilter1_Mmin5.7e+08_RHIImax20_256_300Mpc')
    d1 = b1.box_data
    # ionized = d1 > 0.
    # ionized_labels = measure.label(ionized)
    # R = measure.regionprops(ionized_labels)
    # print 'computing edt'
    # EDT = ndimage.distance_transform_edt(ionized)
    # print 'computing cdt'
    # CDT = ndimage.distance_transform_cdt(ionized)
    # print 'computing max'
    labels, markers, EDT = watershed_3d(d1)
    #import IPython; IPython.embed()
    # print 'computing bdt'
    # BDT = ndimage.distance_transform_bf(1-marker_ionized)
    # import pylab as plt
    plt.figure()
    plt.subplot(141)
    plt.imshow(d1[128])
    plt.subplot(142)
    plt.imshow(EDT[128])
    plt.subplot(143)
    plt.imshow(labels[128])
    plt.subplot(144)
    plt.imshow(markers[128])
    plt.show()