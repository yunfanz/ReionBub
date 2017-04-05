import numpy as np
import os, fnmatch
from sys import argv
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max
from tocmfastpy import *
from h_transform_globalsync import *
import pylab as plt
import seaborn as sns
import ws3d_gpu, edt_cuda
from joblib import Parallel, delayed
from IO_utils import *
import optparse
#possibly useful
#morphology.remove_small_objects

def local_maxima_debug(arr, ionized, threshold_h=0.7, connectivity=2, try_loading=False, outfile='smoothed_11.npy', smoothing='hmax'):

    neighborhood = ndimage.morphology.generate_binary_structure(len(arr.shape), connectivity)
    #maxima = None
    maxima = peak_local_max(arr, labels=ionized, footprint=neighborhood, indices=False, exclude_border=False)
    if smoothing == 'hmax': #smoothing with h-max transform
        if try_loading:
            try:
                print "loading h_max_transform"
                smoothed_arr = np.load('smoothed.npy')
            except: 
                smoothed_arr = h_max_cpu(arr, neighborhood, maxima, threshold_h, mask=ionized, connectivity=connectivity)
                np.save(outfile, smoothed_arr)
        else:
            smoothed_arr = h_max_cpu(arr, neighborhood, maxima, threshold_h, mask=ionized, connectivity=2, max_iterations=5)
            np.save(outfile, smoothed_arr)
        maxima = peak_local_max(smoothed_arr, labels=ionized, footprint=neighborhood, indices=False, exclude_border=False)
    # elif smoothing == 'bin':
    #     print 'Smoothing field with binary dilation'
    #     n_reg = 0
    #     m_reg = 1000
    #     while True:
    #         maxima = ionized & ndimage.binary_dilation(maxima, structure=neighborhood, iterations=1) #smoothing with binary dilation
    #         tmp_labels = measure.label(maxima, connectivity=connectivity)
    #         m_reg = len(measure.regionprops(tmp_labels))
    #         print m_reg
    #         if m_reg == n_reg: break
    #         n_reg = m_reg
    return maxima #np.where(detected_maxima)

def local_maxima_cpu(arr, ionized, threshold_h=0.7, connectivity=2, save=False, outfile='smoothed_11.npy'):
    neighborhood = ndimage.morphology.generate_binary_structure(len(arr.shape), connectivity)
    maxima = peak_local_max(arr, labels=ionized, footprint=neighborhood, indices=False, exclude_border=False)
    if threshold_h > 0:
        arr = h_max_cpu(arr, neighborhood, maxima, threshold_h, mask=ionized, connectivity=2, max_iterations=50)
        maxima = peak_local_max(arr, labels=ionized, footprint=neighborhood, indices=False, exclude_border=False)

    return maxima, arr

def local_maxima_gpu(arr, ionized, threshold_h=0.7, connectivity=2):
    s_arr, maxima = h_max_gpu(arr=arr,mask=ionized, maxima=None, h=threshold_h, n_iter=1000)
    return maxima, s_arr

def watershed_3d(image, connectivity=2, h=0.7, target='cuda', edtfile=None):
    ionized = (image == 1.)
    #ionized = ionized*morphology.remove_small_objects(ionized, 3)  #speeds up later process
    if target == 'cuda' or target == 'gpu':
        print 'Computing EDT'
        EDT = None
        try:
            EDT = np.load(edtfile)['EDT']
        except:
            EDT = ndimage.distance_transform_edt(ionized)
        #EDT_c = edt_cuda.distance_transform_edt(arr=ionized)
        #
        maxima, smEDT = local_maxima_gpu(EDT.copy(), ionized, connectivity=connectivity, threshold_h=h)
        #import IPython; IPython.embed()
        print 'Computing watershed'
        if True:
            labels = ws3d_gpu.watershed(-smEDT, mask=ionized)
            #import IPython; IPython.embed()
            markers = measure.label(maxima, connectivity=connectivity)
        else:
            markers = measure.label(maxima, connectivity=connectivity)
            labels = morphology.watershed(-smEDT, markers, mask=ionized)
            #flabels = morphology.watershed(-smEDT, markers, mask=np.ones_like(ionized))
            import IPython; IPython.embed()
    elif target == 'cpu':
        print 'Computing EDT'
        EDT = ndimage.distance_transform_edt(ionized)
        maxima, smEDT = local_maxima_cpu(EDT.copy(), ionized, connectivity=connectivity, threshold_h=h)
        print 'Computing watershed'
        markers = measure.label(maxima, connectivity=connectivity)
        labels = morphology.watershed(-EDT, markers, mask=ionized)
    return labels, markers, EDT, smEDT

def _get_var(Q, logR):
    R = np.exp(logR)
    return -Q/4/np.pi/R**3
def get_size_dist(labels, Q, scale=1, log=True, n_bins=20):
    R = measure.regionprops(labels)
    R_eff = scale*np.array([r.equivalent_diameter/2 for r in R])
    #R_eff = (3*volumes/4/np.pi)**(1./3)
    #dn/dr*(4pi*r**4/3Q) = dn/d(r**(-3))

    if not log:
        hist,bins = np.histogram(R_eff, normed=True, bins=100)
    else:
        logR = np.log(R_eff)
        # var = -Q/4/np.pi/R_eff**3
        log_edges = np.linspace(np.min(logR)-1,np.max(logR)+1,n_bins)
        #var_edges = _get_var(Q, log_edges)
        hist,bin_edges = np.histogram(logR, bins=log_edges, normed=True)
        bws = (log_edges[1:]-log_edges[:-1])/2
        bins = np.exp((log_edges[1:]+log_edges[:-1])/2)
        hist *= 4*np.pi*bins**3/3/Q
        hist /= np.dot(hist, bws)
        
        #hist = hist/Q*4*np.pi*(bins)**3/3
    return hist, bins

def plot_zscroll_dist(fn1='watershed_z10.npz', fn2='watershed_z11.npz', fn3='watershed_z12.npz'):
    plt.figure()
    for fn in [fn1,fn2,fn3]:
        f = np.load(fn)
        hist, bins = get_size_dist(f['labels'], f['Q'], f['scale'])
        plt.plot(bins, hist, label=fn.split('.')[0].split('_')[1])
    plt.xscale('log')
    plt.xlabel('R(Mpc)')
    plt.ylabel(r'\frac{dP}{d\ln r}')
    plt.legend()
    sns.set_context("talk", font_scale=1.4)


def plot_dist(labels, scale=1):
    """
    scale is Mpc/pixel
    """
    R = measure.regionprops(labels)
    R_eff = scale*np.array([r.equivalent_diameter/2 for r in R])
    logR = np.log(R_eff)
    var = -Q/4/np.pi/R_eff**3
    log_edges = np.linspace(np.min(logR)-1,np.max(logR)+1,n_bins)
    var_edges = _get_var(Q, log_edges)
    hist,bin_edges = np.histogram(var, bins=var_edges, normed=True)
    bins = np.exp((log_edges[1:]+log_edges[:-1])/2)

    sns.distplot(R_eff, hist=False, bins=bins)

def watershed_21cmBox(path):
    box = boxio.readbox(path)
    return watershed_3d(box.box_data)

def mc_test(N=1000,SIZE=200):
    x, y, z = np.indices((SIZE, SIZE,SIZE))
    image = np.zeros_like(x)
    print image.shape
    for n in xrange(N):
        print n
        x1, y1, z1 = np.random.randint(0,SIZE, size=3)
        r1 = np.random.randint(1,SIZE/10)
        mask_circle1 = (x - x1)**2 + (y - y1)**2 + (z - z1)**2< r1**2
        image = np.logical_or(mask_circle1, image)

    distance = ndimage.distance_transform_edt(image)
    # local_maxi = peak_local_max(distance, labels=image,
    #                          footprint=np.ones((3, 3, 3)),
    #                          indices=False)
    # markers = ndimage.label(local_maxi)[0]
    # labels = morphology.watershed(-distance, markers, mask=image)
    sd, maxima = h_max_gpu(arr=distance,mask=image, maxima=None, h=1.0, n_iter=150, connectivity=3)
    labels = ws3d_gpu.watershed(-sd, mask=image)
    markers = measure.label(maxima, connectivity=3)
    flabels = morphology.watershed(-sd, markers, mask=image)
    import matplotlib
    carr = np.random.rand(256, 3); carr[0,:] = 0
    cmap = matplotlib.colors.ListedColormap(carr)
    plt.subplot(121)
    plt.imshow(labels[SIZE/2], cmap=cmap)
    plt.subplot(122)
    plt.imshow(flabels[SIZE/2], cmap=cmap)
    import IPython; IPython.embed()

def circle_test():
    x, y, z = np.indices((80, 80,80))
    x1, y1, z1, x2, y2, z2 = 28, 28,50, 44, 52,54
    r1, r2 = 26, 40
    mask_circle1 = (x - x1)**2 + (y - y1)**2 + (z - z1)**2< r1**2
    mask_circle2 = (x - x2)**2 + (y - y2)**2 + (z - z2)**2< r2**2
    image = np.logical_or(mask_circle1, mask_circle2)
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance
    # to the background
    distance = ndimage.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, labels=image,
                             footprint=np.ones((3, 3, 3)),
                             indices=False)
    markers = ndimage.label(local_maxi)[0]
    #labels = morphology.watershed(-distance, markers, mask=image)
    flabels = ws3d_gpu.watershed(-distance, mask=image)
    import matplotlib
    carr = np.random.rand(256, 3); carr[0,:] = 0
    cmap = matplotlib.colors.ListedColormap(carr)
    fig, axes = plt.subplots(1,1)
    #axes[0].imshow(labels[40],cmap=cmap)
    axes.imshow(flabels[40],cmap=cmap)
    import IPython; IPython.embed()




if __name__ == '__main__':

    o = optparse.OptionParser()
    o.add_option('-d','--dir', dest='DIR', default='/home/yunfanz/Data/21cmFast/Boxes/')
    o.add_option('-o','--out', dest='OUTDIR', default='./NPZ/')
    
    files = find_files(DIR, pattern='xH_nohalos_z012*')
    

    for path in [files[0]]:
        print 'Processing', path
        b1 = boxio.readbox(path)
        d1 = 1 - b1.box_data
        #d1 = 1 - b1.box_data#[:252,:252,:252]
        scale = float(b1.param_dict['dim']/b1.param_dict['BoxSize'])
        #OUTFILE = b1.param_dict['basedir']+'/watershed_z{0}.npz'.format(b1.z)
        OUTFILE = OUTDIR+'dwatershed_z{0}_L{1}_Iter{2}.npz'.format(b1.z, b1.param_dict['BoxSize'], b1.param_dict['Iteration'])
        labels, markers, EDT, smEDT = watershed_3d(d1, h=0.35, target='gpu', connectivity=3, edtfile=OUTFILE)
        Q_a = 1 - b1.param_dict['nf']
        print 'Q', Q_a
        print 'saving', OUTFILE
        np.savez(OUTFILE, Q=Q_a, scale=scale, labels=labels, markers=markers, EDT=EDT, smEDT=smEDT)


    #hist, bins = get_size_dist(labels, Q, scale=scale)

    # import matplotlib
    # carr = np.random.rand(256, 3); carr[0,:] = 0
    # cmap = matplotlib.colors.ListedColormap(carr)

    # import IPython; IPython.embed()

