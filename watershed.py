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

def local_maxima(arr, ionized, threshold_h=0.7, connectivity=2, try_loading=False):

    neighborhood = ndimage.morphology.generate_binary_structure(len(arr.shape), connectivity)
    maxima = peak_local_max(arr, labels=ionized, footprint=neighborhood, indices=False, exclude_border=False)
    if try_loading:
        try:
            print "loading h_max_transform"
            smoothed_arr = np.load('smoothed.npy')
        except: 
            smoothed_arr = h_max_transform(arr, neighborhood, maxima, threshold_h, mask=ionized, connectivity=connectivity)
            np.save('smoothed.npy', smoothed_arr)
    else:
        smoothed_arr = h_max_transform(arr, neighborhood, maxima, threshold_h, mask=ionized, connectivity=connectivity)
        np.save('smoothed.npy', smoothed_arr)
    maxima = peak_local_max(smoothed_arr, labels=ionized, footprint=neighborhood, indices=False, exclude_border=False)
    return maxima #np.where(detected_maxima)


def watershed_3d(image, connectivity=2):
    ionized = image > 0.998
    Q = np.sum(ionized).astype(np.float32)/image.size #naive filling fraction
    ionized = ionized*morphology.remove_small_objects(ionized, 3)  #speeds up later process
    EDT = ndimage.distance_transform_edt(ionized)
    maxima = local_maxima(EDT.copy(), ionized, connectivity=connectivity)
    markers = measure.label(maxima, connectivity=connectivity)
    labels = morphology.watershed(-EDT, markers, mask=ionized)
    
    return labels, markers, EDT, Q

def get_size_dist(labels, Q, scale=1, log=True):
    R = measure.regionprops(labels)
    R_eff = scale*np.array([r.equivalent_diameter/2 for r in R])
    #R_eff = (3*volumes/4/np.pi)**(1./3)

    if not log:
        hist,bins = np.histogram(R_eff, normed=True, bins=100)
    else:
        logR = np.log(R_eff)
        bin_edges = np.linspace(np.min(logR),np.max(logR),100)
        hist,_ = np.histogram(R_eff, bins=bin_edges, normed=True)
        bins = np.exp((bin_edges[1:]+bin_edges[:-1])/2)
        hist = hist/Q*4*np.pi*(bins)**3/3
    return hist, bins

def plot_dist(labels, scale=1):
    """
    scale is Mpc/pixel
    """
    R = measure.regionprops(labels)
    R_eff = scale*np.array([r.equivalent_diameter/2 for r in R])
    bins = np.logspace(-1,2.,100)
    sns.distplot(R_eff, hist=False, bins=bins)

def watershed_21cmBox(path):
    box = boxio.readbox(path)
    return watershed_3d(box.box_data)

if __name__ == '__main__':
    #b1 = boxio.readbox('../pkgs/21cmFAST/TrialBoxes/xH_nohalos_z008.06_nf0.604669_eff20.0_effPLindex0.0_HIIfilter1_Mmin5.7e+08_RHIImax20_256_300Mpc')
    #b1 = boxio.readbox('../pkgs/21cmFAST/Boxes/xH_nohalos_z010.00_nf0.865885_eff20.0_effPLindex0.0_HIIfilter1_Mmin4.3e+08_RHIImax20_500_500Mpc')
    #DIR = '../pkgs/21cmFAST/Boxes/'
    #FILE = 'xH_nohalos_z010.00_nf0.873649_eff20.0_effPLindex0.0_HIIfilter1_Mmin4.3e+08_RHIImax30_500_250Mpc'
    DIR = '/data2/21cmFast/lin0_logz10-35_box500_dim500/Boxes/'
    FILE = 'xH_nohalos_z010.00_nf0.865885_eff20.0_effPLindex0.0_HIIfilter1_Mmin4.3e+08_RHIImax20_500_500Mpc'
    PATH = DIR+FILE
    #PATH = '/home/yunfanz/Data/21cmFast/Boxes/xH_nohalos_z010.00_nf0.881153_eff20.0_effPLindex0.0_HIIfilter1_Mmin4.3e+08_RHIImax20_400_100Mpc'
    b1 = boxio.readbox(PATH)
    d1 = b1.box_data#[::5,::5,::5]
    scale = float(b1.param_dict['BoxSize'])/b1.param_dict['dim']
    labels, markers, EDT, Q = watershed_3d(d1)
    OUTFILE = b1.param_dict['basedir']+'/watershed_z'+str(int(np.round(b1.z)))+'.npz'
    print 'saving', OUTFILE
    np.savez(OUTFILE, Q=Q, scale=scale, labels=labels, markers=markers, EDT=EDT)


    hist, bins = get_size_dist(labels, Q, scale=scale)
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