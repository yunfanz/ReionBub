import numpy as np
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max
from tocmfastpy import *
import pylab as plt

#possibly useful
#morphology.remove_small_objects

def h_max_transform(arr, neighborhood, mask, h, max_iterations=50):
    """
    Brute force function to compute hMaximum smoothing
    arr: values such as EDT
    neighborhood: structure to step connected regions
    mask: maxima of arr
    """
    print "Starting brute force h-transform, max_iteration", max_iterations
    tmp_arr = arr
    tmp_labels = measure.label(mask, connectivity=3) #con should be 2 for face, 3 for edge or corner 
    i = 0 
    while i<max_iterations:
        newmask = ndimage.binary_dilation(mask, structure=neighborhood)
        diff = ndimage.filters.maximum_filter(tmp_arr, footprint=neighborhood) - tmp_arr
        newmask = newmask & (diff < h)
        if not (newmask ^ mask).any(): 
            print 'h_transform completed in iteration', i
            break
        tmp_labels = measure.label(newmask, connectivity=3)
        L = len(measure.regionprops(tmp_labels, intensity_image=arr))
        print 'iteration', i, 'number of regions', L
        for region in measure.regionprops(tmp_labels, intensity_image=arr):
            tmp_arr[np.where(region.image)] = region.max_intensity 
            #also see ndimage.labeled_comprehension
        mask = newmask
        i += 1
    #import IPython; IPython.embed()
    for region in measure.regionprops(tmp_labels, intensity_image=arr):
        tmp_arr[np.where(region.image)] = region.max_intensity - h
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
    ionized = image > 0.99
    ionized = ionized*morphology.remove_small_objects(ionized, 3)  #speeds up later process
    EDT = ndimage.distance_transform_edt(ionized)
    maxima = local_maxima(EDT.copy(), ionized)
    markers = ndimage.label(maxima)[0]
    labels = morphology.watershed(-EDT, markers, mask=ionized)
    
    return labels, markers, EDT

def get_size_dist(labels):
    R = measure.regionprops(labels)
    R_eff = np.array([r.equivalent_diameter/2 for r in R])

    #R_eff = (3*volumes/4/np.pi)**(1./3)
    hist,bins = np.histogram(R_eff, normed=True, bins=100)
    loghist = hist*bins[1:]
    return loghist, bins
def plot_dist(labels):
    R = measure.regionprops(labels)
    R_eff = np.array([r.equivalent_diameter/2 for r in R])
    bins = np.logspace(0,2.6,100)
    sns.plot_dist(R_eff, hist=False, bins=bins)

def watershed_21cmBox(path):
    box = boxio.readbox(path)
    return watershed_3d(box.box_data)

if __name__ == '__main__':
    #b1 = boxio.readbox('../pkgs/21cmFAST/TrialBoxes/xH_nohalos_z008.06_nf0.604669_eff20.0_effPLindex0.0_HIIfilter1_Mmin5.7e+08_RHIImax20_256_300Mpc')
    b1 = boxio.readbox('../pkgs/21cmFAST/Boxes/xH_nohalos_z010.00_nf0.865885_eff20.0_effPLindex0.0_HIIfilter1_Mmin4.3e+08_RHIImax20_500_500Mpc')
    d1 = b1.box_data#[::5,::5,::5]
    # ionized = d1 > 0.
    # ionized_labels = measure.label(ionized)
    # R = measure.regionprops(ionized_labels)
    # print 'computing edt'
    # EDT = ndimage.distance_transform_edt(ionized)
    # print 'computing cdt'
    # CDT = ndimage.distance_transform_cdt(ionized)
    # print 'computing max'
    labels, markers, EDT = watershed_3d(d1)

    #hist, bins = get_size_dist(labels)
    import IPython; IPython.embed()
    # print 'computing bdt'
    # BDT = ndimage.distance_transform_bf(1-marker_ionized)
    # import pylab as plt
    # plt.figure()
    # plt.subplot(141)
    # plt.imshow(d1[128])
    # plt.subplot(142)
    # plt.imshow(EDT[128])
    # plt.subplot(143)
    # plt.imshow(labels[128])
    # plt.subplot(144)
    # plt.imshow(markers[128])

    # plt.figure()
    # plt.plot(bins[1:], hist)
    # plt.show()