import numpy as np
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max

# H-max transform Accelerated with PyCUDA
# Yunfan Zhang
# yf.g.zhang@gmail.com
# 3/09/2017
# Usage: python GameOfLife.py n n_iter
# where n is the board size and n_iter the number of iterations
import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import sys
import numpy as np
from pylab import cm as cm
import matplotlib.pyplot as plt

def random_init(n):
    #np.random.seed(100)
    M = np.zeros((n,n,n)).astype(np.int32)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                M[k,j,i] = np.int32(np.random.randint(2))
    return M

kernel_code_template = """
  __global__ void step(float *C, float *M, bool *Mask, bool *maxima, float h, int conn)
  {
    bool ismax = true;
    //int n_x = blockDim.x*gridDim.x;
    //int n_y = blockDim.y*gridDim.y;
    int n_x = %(NDIM)s; 
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    int k = threadIdx.z + blockDim.z*blockIdx.z;
    if ( ( i < %(NDIM)s ) && ( j < %(NDIM)s ) && ( k < %(NDIM)s ) )
    {
        int threadId = k*n_x*n_x + j*n_x + i;
        int i_left; int i_right; int j_down; int j_up; int k_down; int k_up;
        //Mirror boundary condition
        if(i==0) {i_left=i;} else {i_left=i-1;}   
        if(i==n_x-1) {i_right=i;} else {i_right=i+1;}
        if(j==0) {j_down=j;} else {j_down=j-1;}
        if(j==n_x-1) {j_up=j;} else {j_up=j+1;}
        if(k==0) {k_down=k;} else {k_down=k-1;}
        if(k==n_x-1) {k_up=k;} else {k_up=k+1;}
        int n_neigh; 
        int* neighbors = NULL;

        switch (conn) {
          case 1:
            n_neigh = 6;
            int neigh1 [6] = {k*n_x*n_x+j*n_x+i_left, k*n_x*n_x+j_down*n_x+i, k*n_x*n_x+j*n_x+i_right,
                                k*n_x*n_x+j_up*n_x+i, k_down*n_x*n_x+j*n_x+i, k_up*n_x*n_x+j*n_x+i};
            neighbors = neigh1;
            break;
          case 2:
            n_neigh = 18;
            int neigh2 [18] = {k*n_x*n_x+j*n_x+i_left, k*n_x*n_x+j_down*n_x+i, k*n_x*n_x+j*n_x+i_right,
                                k*n_x*n_x+j_up*n_x+i, k_down*n_x*n_x+j*n_x+i, k_up*n_x*n_x+j*n_x+i,
                                k*n_x*n_x+j_up*n_x+i_left, k*n_x*n_x+j_down*n_x+i_left, k*n_x*n_x+j_down*n_x+i_right,
                                k*n_x*n_x+j_up*n_x+i_right, k_down*n_x*n_x+j_down*n_x+i, k_up*n_x*n_x+j_down*n_x+i,
                                k_down*n_x*n_x+j*n_x+i_left, k_down*n_x*n_x+j_up*n_x+i, k_up*n_x*n_x+j_up*n_x+i,
                                k_down*n_x*n_x+j*n_x+i_right, k_up*n_x*n_x+j*n_x+i_left, k_up*n_x*n_x+j*n_x+i_right};
            neighbors = neigh2;
            break;
          default:
            n_neigh = 18;
            int neighd [18] = {k*n_x*n_x+j*n_x+i_left, k*n_x*n_x+j_down*n_x+i, k*n_x*n_x+j*n_x+i_right,
                                k*n_x*n_x+j_up*n_x+i, k_down*n_x*n_x+j*n_x+i, k_up*n_x*n_x+j*n_x+i,
                                k*n_x*n_x+j_up*n_x+i_left, k*n_x*n_x+j_down*n_x+i_left, k*n_x*n_x+j_down*n_x+i_right,
                                k*n_x*n_x+j_up*n_x+i_right, k_down*n_x*n_x+j_down*n_x+i, k_up*n_x*n_x+j_down*n_x+i,
                                k_down*n_x*n_x+j*n_x+i_left, k_down*n_x*n_x+j_up*n_x+i, k_up*n_x*n_x+j_up*n_x+i,
                                k_down*n_x*n_x+j*n_x+i_right, k_up*n_x*n_x+j*n_x+i_left, k_up*n_x*n_x+j*n_x+i_right};
            neighbors = neighd; 
          }
        
        int ne;

        if (!Mask[threadId]) {ismax = false;}
        else
        {
          for (int ni=0; ni<n_neigh; ni++) 
          {
            ne = neighbors[ni];
            if (C[threadId]<C[ne]) {ismax = false;}
          }
        }
        maxima[threadId] = ismax;
        M[threadId] = C[threadId];
        __syncthreads();

        if (Mask[threadId])
        {
          for (int ni=0; ni<n_neigh; ni++) 
          {
            ne = neighbors[ni];
            if ( (maxima[ne]) && (C[threadId] >= C[ne] - h) ) 
            {
                M[threadId] = (C[threadId]<C[ne]) ? C[ne] : C[threadId];
            }
          }
        }
    }
  }
"""

def h_max_gpu(filename=None, arr=None, mask=None, maxima=None, h=0.7, n_iter=50, n_block=8):
    DRAW = False
    if filename is not None:
        file = np.load(filename)
        arr = file['arr']; mask = file['mask']
    if arr is None: raise Exception('No input specified!')
    # arr = random_init(50)
    if mask is None: mask = arr > 0
    if maxima is None: maxima = arr > 0
    arr = arr.astype(np.float32)
    M = arr.copy()
    n = arr.shape[0]
    n_grid = int(np.ceil(float(n)/n_block))
    print n_grid
    #n = n_block*n_grid
    kernel_code = kernel_code_template % {
        'NDIM': n
    }
    mod = SourceModule(kernel_code)
    func = mod.get_function("step")
    print "Tranferring data to gpu"
    C_gpu = gpuarray.to_gpu( arr )
    M_gpu = gpuarray.to_gpu( M )
    mask_gpu = gpuarray.to_gpu( mask )
    max_gpu = gpuarray.to_gpu( maxima )
    h_gpu = gpuarray.to_gpu(np.array(0.7, dtype=np.float32))
    conn_gpu = gpuarray.to_gpu(np.array(2, dtype=np.int32))
    # print(h_gpu.get())
    print "Starting gpu force h-transform with iteration", n_iter
    if DRAW:
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        fig.suptitle("H_max")
        ax.set_title('Number of Iterations = %d'%(n_iter))
        myobj = plt.imshow(C_gpu.get()[8],origin='lower',cmap='Greys',  interpolation='nearest')
        plt.ion()
        plt.draw()
    for k in range(n_iter):
        func(C_gpu,M_gpu,mask_gpu, max_gpu, h_gpu, conn_gpu, block=(n_block,n_block,n_block),grid=(n_grid,n_grid,n_grid))
        C_gpu, M_gpu = M_gpu, C_gpu
        if DRAW:
            myobj.set_data(C_gpu.get()[8])
            ax.set_title('Number of Iterations = %d'%(k))
            plt.draw()
    arr_transformed = C_gpu.get()
    maxima_trans = max_gpu.get()
    print "exiting h_max_gpu"
    return arr_transformed, maxima_trans

# if __name__=='__main__':
#     n = int(sys.argv[1])
#     n_iter = int(sys.argv[2])
#     M = h_transform()
#     print(M.shape)
# print("%d live cells after %d iterations" %(np.sum(C_gpu.get()),n_iter))
# fig = plt.figure(figsize=(12,12))
# ax = fig.add_subplot(111)
# fig.suptitle("Conway's Game of Life Accelerated with PyCUDA")
# ax.set_title('Number of Iterations = %d'%(n_iter))
# myobj = plt.imshow(C_gpu.get()[8],origin='lower',cmap='Greys',  interpolation='nearest',vmin=0, vmax=1)
# plt.pause(.01)
# plt.draw()
# m = 0
# while m <= n_iter:
#     m += 1
#     func(C_gpu,M_gpu,block=(n_block,n_block,n_block),grid=(n_grid,n_grid,n_grid))
#     C_gpu, M_gpu = M_gpu, C_gpu
#     myobj.set_data(C_gpu.get()[8])
#     ax.set_title('Number of Iterations = %d'%(m))
#     plt.pause(.01)
#     plt.draw()

def h_max_cpu(arr, neighborhood, markers, h, mask=None, connectivity=2, max_iterations=50):
    """
    Brute force function to compute hMaximum smoothing
    arr: values such as EDT
    neighborhood: structure to step connected regions
    markers: maxima of arr
    """
    tmp_arr = arr.copy()
    arrshape = arr.shape
    tmp_labels = measure.label((mask & markers), connectivity=connectivity) #con should be 2 for face, 3 for edge or corner 
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
            tmp_arr[coord[0], coord[1], coord[2]] = region.max_intensity - h
        else:
            tmp_arr[coord[0], coord[1]] = region.max_intensity - h
    return tmp_arr