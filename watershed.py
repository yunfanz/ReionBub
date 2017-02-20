import numpy as np
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max
from tocmfastpy import *
import pylab as plt
from numba import jit, vectorize, guvectorize, autojit
import seaborn as sns
#possibly useful
#morphology.remove_small_objects
#@autojit
# @guvectorize('void(float32[:], int32[:,:], float32[:])', '(m),(l,n),(l)')
# def update_region(arr, coords, values):
#     L = coords.shape[0]
#     for i in range(L):  
#         #for coordinate in coords[i]:     
#             #arr[coordinates[0], coordinates[1]] = values[i]         
#         arr[coordinates[i]] = values[i]



def h_max_transform(arr, neighborhood, markers, h, mask=None, connectivity=2, max_iterations=50):
    """
    Brute force function to compute hMaximum smoothing
    arr: values such as EDT
    neighborhood: structure to step connected regions
    markers: maxima of arr
    """
    tmp_arr = arr.copy()
    arrshape = arr.shape
    tmp_labels = measure.label(markers, connectivity=connectivity) #con should be 2 for face, 3 for edge or corner 
    L = len(measure.regionprops(tmp_labels, intensity_image=arr))
    print "Starting brute force h-transform, max_iteration", max_iterations, 'initial regions', L
    i = 0 
    while i<max_iterations:
        newmarkers = mask & ndimage.binary_dilation(markers, structure=neighborhood)
        diff = ndimage.filters.maximum_filter(tmp_arr, footprint=neighborhood) - tmp_arr
        newmarkers = newmarkers & (diff <= h)
        if not (newmarkers ^ markers).any(): 
            print 'h_transform completed in iteration', i
            break
        tmp_labels = measure.label(newmarkers, connectivity=connectivity)
        L = len(measure.regionprops(tmp_labels, intensity_image=arr))
        print 'iteration', i, 'number of regions', L

        for region in measure.regionprops(tmp_labels, intensity_image=arr):
            #tmp_arr[np.where(region.image)] = region.max_intensity 
            coord = region.coords.T
            assert coord.shape[0] <= 3
            if coord.shape[0] == 3:
                tmp_arr[coord[0], coord[1], coord[2]] = region.max_intensity 
            else:
                tmp_arr[coord[0], coord[1]] = region.max_intensity
            #also see ndimage.labeled_comprehension
        print 'next'

        markers = newmarkers
        i += 1
    for region in measure.regionprops(tmp_labels, intensity_image=arr):
        #tmp_arr[np.where(region.image)] = region.max_intensity 
        coord = region.coords.T
        if coord.shape[0] == 3:
            tmp_arr[coord[0], coord[1], coord[2]] = region.max_intensity 
        else:
            tmp_arr[coord[0], coord[1]] = region.max_intensity - h
    return tmp_arr

# def unique_local_max(arr, labels, footprint):
#     maxima = peak_local_max(arr=arr, labels=labels, footprint=footprint, indices=False, exclude_border=False)
#     R = measure.regionprops(maxima)
#     for r in R:
#         if r.area > 1:

def local_maxima(arr, ionized, threshold_h=0.3, connectivity=2, save=None):

    neighborhood = ndimage.morphology.generate_binary_structure(len(arr.shape), connectivity)
    maxima = peak_local_max(arr, labels=ionized, footprint=neighborhood, indices=False, exclude_border=False)
    if threshold_h: 
        smoothed_arr = h_max_transform(arr, neighborhood, maxima, threshold_h, mask=ionized, connectivity=connectivity)
        maxima = peak_local_max(smoothed_arr, labels=ionized, footprint=neighborhood, indices=False, exclude_border=False)
    else:
        print "Skipping h_max_transform"
    if True:
        np.save('smoothed.npy', smoothed_arr)
    return maxima #np.where(detected_maxima)


def watershed_3d(image, connectivity=2):
    ionized = image > 0.997
    ionized = ionized*morphology.remove_small_objects(ionized, 3)  #speeds up later process
    EDT = ndimage.distance_transform_edt(ionized)
    maxima = local_maxima(EDT.copy(), ionized, connectivity=connectivity)
    markers = measure.label(maxima, connectivity=connectivity)
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
    bins = np.logspace(0,2.,100)
    sns.distplot(R_eff, hist=False, bins=bins)

def watershed_21cmBox(path):
    box = boxio.readbox(path)
    return watershed_3d(box.box_data)

if __name__ == '__main__':
    #b1 = boxio.readbox('../pkgs/21cmFAST/TrialBoxes/xH_nohalos_z008.06_nf0.604669_eff20.0_effPLindex0.0_HIIfilter1_Mmin5.7e+08_RHIImax20_256_300Mpc')
    #b1 = boxio.readbox('../pkgs/21cmFAST/Boxes/xH_nohalos_z010.00_nf0.865885_eff20.0_effPLindex0.0_HIIfilter1_Mmin4.3e+08_RHIImax20_500_500Mpc')
    PATH = '/home/yunfanz/Data/21cmFast/Boxes/xH_nohalos_z010.00_nf0.865885_eff20.0_effPLindex0.0_HIIfilter1_Mmin4.3e+08_RHIImax20_500_500Mpc'
    b1 = boxio.readbox(PATH)
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