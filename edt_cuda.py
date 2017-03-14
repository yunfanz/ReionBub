import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import sys
import numpy as np

template = """
__global__ void EDT_kernel(const bool* img, float* dist)
{
    int n_x = %(NDIM)s; 
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    int k = threadIdx.z + blockDim.z*blockIdx.z;
    if ( ( i >= %(NDIM)s ) || ( j >= %(NDIM)s ) && ( k >= %(NDIM)s ) )
    {
        return;
    }
 
    int threadId = k*n_x*n_x + j*n_x + i;
    int N = n_x*n_x*n_x;

    float minv = INFINITY;

    if (img[threadId] > 0)
    {
        minv = 0.0f;
    }
    else
    {
        for (int ind = 0; ind < N; ind++)
        {
            if (img[ind] > 0)
            {
              
              int z = ind / n_x / n_x;
              int y = (ind - z*n_x*n_x) / n_x;
              int x = (ind - z*n_x*n_x - y*n_x);
              //float d = sqrtf( powf(float(x-i), 2.0f) + powf(float(y-j), 2.0f) + powf(float(z-k), 2.0f));
              float d = 0;
              if (d < minv) minv = d;
            }
        }
    }
    dist[i] = minv;
}
"""

def distance_transform_edt(arr=None, filename=None, n_block=8):
    if filename is not None:
        file = np.load(filename)
        arr = file['arr']
    if arr is None: raise Exception('No input specified!')
    try:
    	arr = arr.astype(bool)
    except:
    	print "input array not binary"
    	arr = arr > 0
    EDT_h = np.zeros(arr.shape, dtype=np.float32)
    n = arr.shape[0]
    print n
    n_grid = int(np.ceil(float(n)/n_block))
    #print n_grid
    #n = n_block*n_grid
    kernel_code = template % {'NDIM': n }
    mod = SourceModule(kernel_code)
    func1 = mod.get_function("EDT_kernel")

    C_gpu = gpuarray.to_gpu( arr )
    EDT_gpu = gpuarray.to_gpu( EDT_h )
    # conn_gpu = gpuarray.to_gpu(np.array(2, dtype=np.int32))
    # print(h_gpu.get())
    print "Starting PyCUDA EDT"

    start = pycuda.driver.Event()
    end = pycuda.driver.Event()
    start.record()
    func1(C_gpu, EDT_gpu, block=(n_block,n_block,n_block),grid=(n_grid,n_grid,n_grid))
    end.record()
    end.synchronize()

    EDT_h = EDT_gpu.get()

    return EDT_h