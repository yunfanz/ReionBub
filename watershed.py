import numpy as np
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max
from tocmfastpy import *


#possibly useful
#morphology.remove_small_objects

def h_max_transform(arr, neighborhood, mask, h, max_iterations=20):
    """
    Brute force function to compute hMaximum smoothing
    arr: values such as EDT
    neighborhood: structure to step connected regions
    mask: maxima of arr
    """
    tmp_arr = arr
    tmp_labels = measure.label(mask, connectivity=2)
    i = 0
    while i<max_iterations:
        newmask = ndimage.binary_dilation(mask, structure=neighborhood)
        diff = ndimage.filters.maximum_filter(tmp_arr, footprint=neighborhood) - tmp_arr
        newmask = newmask & (diff < h)
        if not (newmask ^ mask).any(): 
            print 'h_transform completed in iteration', i
            break
        tmp_labels = measure.label(newmask, connectivity=2)
        for region in measure.regionprops(tmp_labels, intensity_image=arr):
            tmp_arr[(tmp_labels==region.label)] = region.max_intensity 
        mask = newmask
        i += 1
    for region in measure.regionprops(tmp_labels, intensity_image=arr):
        tmp_arr[(tmp_labels==region.label)] = region.max_intensity - h
    return tmp_arr


def local_maxima(arr, ionized, threshold_h=0.9, connectivity=2):

    neighborhood = ndimage.morphology.generate_binary_structure(len(arr.shape), connectivity)
    maxima = peak_local_max(arr, labels=ionized, footprint=neighborhood, indices=False)
    if threshold_h: 
        smoothed_arr = h_max_transform(arr, neighborhood, maxima, threshold_h)
        maxima = peak_local_max(smoothed_arr, labels=ionized, footprint=neighborhood, indices=False)
    else:
        print "Skipping h_max_transform"
    return maxima #np.where(detected_maxima)


def watershed_3d(image):
    ionized = image > 0.
    ionized = ionized*morphology.remove_small_objects(ionized, 6)
    EDT = ndimage.distance_transform_edt(ionized)
    maxima = local_maxima(EDT, ionized)
    markers = ndimage.label(maxima)[0]
    labels = morphology.watershed(-EDT, markers, mask=ionized)
    
    return labels, markers, EDT

def watershed_21cmBox(path):
    box = boxio.readbox(path)
    return watershed_3d(box.box_data)

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