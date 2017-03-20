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
#define INF 9999999999
// Convert 3D index to 1D index.
#define INDEX(k,j,i,ld) ((k)*ld*ld + (j) * ld + (i))
#define EPSILON 0.0000002 //tolerance for machine precision

// Convert local (shared memory) coord to global (image) coordinate.
#define L2I(ind,off) (((ind) / blockDim.x) * (blockDim.x - 2) + (off)-1)

__constant__ int N1_xs[6] = {0,1,1,-1,-1,0};
__constant__ int N1_ys[6] = {0,1,-1,-1,1,0};
__constant__ int N1_zs[6] = {-1,0,0,0,0,1};

__constant__ int N2_xs[18] = {0,0,1,0,-1,-1,0,1,1,1,0,-1,-1,0,1,0,-1,0};
__constant__ int N2_ys[18] = {0,-1,0,1,0,-1,-1,-1,0,1,1,1,0,-1,0,1,0,0};
__constant__ int N2_zs[18] = {-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1};

__constant__ int N3_xs[26] = {0,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,-1,-1,0};
__constant__ int N3_ys[26] = {0,-1,-1,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,0};
__constant__ int N3_zs[26] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1};

__global__ void step(float *C, float *M, bool *Mask, bool *maxima)
{
    bool ismax = true;
    int w = %(NDIM)s; 
    int bsize = blockDim.x - 2;
    int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
    int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
    int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
    int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
    int img_x = L2I(i,tx);
    int img_y = L2I(j,ty);
    int img_z = L2I(k,tz);
    int new_w = w + w/(bdx-2)*2+1;
    //int new_w = w + w * 2;
    //int threadId = INDEX(k,j,i, new_w);
    int p = INDEX(img_z,img_y,img_x,%(NDIM)s);
    int bp = INDEX(tz,ty,tx,%(BLOCKS)s);
    int ghost = (tx == 0 || ty == 0 || tz == 0 || tx == bdx - 1 || ty == bdy - 1 || tz == bdz - 1);

    //Mirror boundary condition
    int px; int py; int pz;
    if(img_x == -1) { px = 0;} 
    else if(img_x == w) {px = w-1;}
    else {px = img_x;}
    if(img_y == -1) { py = 0;} 
    else if(img_y == w) {py = w-1;}
    else {py = img_y;}
    if(img_z == -1) { pz = 0;} 
    else if(img_z == w) {pz = w-1;}
    else {pz = img_z;}
    int pp = INDEX(pz,py,px,%(NDIM)s);
    __shared__ float s_C[%(BLOCKS)s*%(BLOCKS)s*%(BLOCKS)s];
    __shared__ float s_MAX[%(BLOCKS)s*%(BLOCKS)s*%(BLOCKS)s];
    s_C[bp] = C[pp];

    __syncthreads();


    if ( ( i < new_w) && ( j < new_w ) && ( k < new_w ) && ghost==0 )
    {

        int n_neigh; 
        int* neigh_xs = NULL; int* neigh_ys = NULL; int* neigh_zs = NULL;

        switch (%(CONN)s) {
          case 1:
            n_neigh = 6;
            neigh_xs = N1_xs; neigh_ys = N1_ys; neigh_zs = N1_zs;
            break;
          case 2:
            n_neigh = 18;
            neigh_xs = N2_xs; neigh_ys = N2_ys; neigh_zs = N2_zs;
            break;
          case 3:
            n_neigh = 26;
            neigh_xs = N3_xs; neigh_ys = N3_ys; neigh_zs = N3_zs;
            break;
        }

        int ne;

        if (!Mask[p]) {ismax = false;}
        else
        {
          for (int ni=0; ni<n_neigh; ni++) 
          {
            int x = neigh_xs[ni]; int y = neigh_ys[ni]; int z = neigh_zs[ni];
            int nex = x+tx; int ney = y+ty; int nez = z+tz;  //shared memory indices of neighbors
            if (s_C[bp] < s_C[INDEX(nez,ney,nex,bdx)]) {ismax = false;}
          }
        }
        maxima[p] = ismax;
        s_MAX[bp] = ismax;
        M[p] = s_C[bp];;

        __syncthreads();

        if (Mask[p])
        {
          for (int ni=0; ni<n_neigh; ni++) 
          {
            
            int x = neigh_xs[ni]; int y = neigh_ys[ni]; int z = neigh_zs[ni];
            int nex = x+tx; int ney = y+ty; int nez = z+tz;  //shared memory indices of neighbors within block
            ne = INDEX(nez,ney,nex,%(BLOCKS)s);
            int h = %(HVAL)s;
            if ( (s_MAX[ne]) && (s_C[bp] < s_C[ne]) && (s_C[bp] > s_C[ne] - h) ) 
            {
                M[p] = s_C[ne];
                //M[p] = ((s_C[bp]<s_C[ne]) && (s_C[bp] > s_C[ne] - h)) ? s_C[ne] : s_C[bp];
            }
          }
        }
    }
}
__global__ void finalize(const float *C, float *M, bool *Mask, bool *maxima)
{
    int w = %(NDIM)s; 
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    int k = threadIdx.z + blockDim.z*blockIdx.z;
    if ( ( i < %(NDIM)s ) && ( j < %(NDIM)s ) && ( k < %(NDIM)s ) )
    {
        int threadId = k*w*w + j*w + i;
        
        if (maxima[threadId])
        {
            M[threadId] = C[threadId]- %(HVAL)s;
        }
    }
}
"""

def h_max_gpu(filename=None, arr=None, mask=None, maxima=None, h=0.7, connectivity=2,n_iter=50, n_block=7):
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
    n_grid = int(np.ceil(float(n)/(n_block-2)))
    print n_grid
    #n = n_block*n_grid
    kernel_code = kernel_code_template % {
        'NDIM': n,
        'HVAL': h,
        'CONN': connectivity,
        'BLOCKS': n_block
    }
    mod = SourceModule(kernel_code)
    func1 = mod.get_function("step")
    func3 = mod.get_function("finalize")

    C_gpu = gpuarray.to_gpu( arr )
    M_gpu = gpuarray.to_gpu( M )
    mask_gpu = gpuarray.to_gpu( mask )
    max_gpu = gpuarray.to_gpu( maxima )
    # conn_gpu = gpuarray.to_gpu(np.array(2, dtype=np.int32))
    # print(h_gpu.get())
    print "Starting PyCUDA h-transform with iteration", n_iter
    
    for k in range(n_iter):
        start = pycuda.driver.Event()
        end = pycuda.driver.Event()
        start.record()
        func1(C_gpu,M_gpu,mask_gpu, max_gpu, block=(n_block,n_block,n_block),grid=(n_grid,n_grid,n_grid))
        end.record()
        end.synchronize()
        C_gpu, M_gpu = M_gpu, C_gpu
        if False:  #For monitoring convergence
            C_cpu = C_gpu.get(); M_cpu = M_gpu.get()
            print "iteration and number of cells changed: ", k, np.sum(np.abs(C_cpu-M_cpu)>0)
    #func3(C_gpu,M_gpu,mask_gpu, max_gpu, block=(n_block,n_block,n_block),grid=(n_grid,n_grid,n_grid))
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