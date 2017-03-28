import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit
from sys import argv
from ws_utils import *

# Read and compile CUDA kernels.
print "Compiling CUDA kernels..."


# PyCUDA wrapper for watershed.
def watershed(I, mask=None):
  kernel_source = open("Dwatershed.cu").read()
  main_module = nvcc.SourceModule(kernel_source)
  descent_kernel = main_module.get_function("descent_kernel")
  stabilize_kernel = main_module.get_function("stabilize_kernel")
  image_texture = main_module.get_texref("img")
  plateau_kernel = main_module.get_function("plateau_kernel")
  minima_kernel = main_module.get_function("minima_kernel")
  flood_kernel = main_module.get_function("flood_kernel")
  increment_kernel = main_module.get_function("increment_kernel")

  # Get contiguous image + shape.
  height, width, depth = I.shape
  I = np.float32(I.copy())
  if mask is None:
    mask = np.ones(I.shape)
  mask = np.int32(mask)

  # Get block/grid size for steps 1-3.
  block_size =  (8,8,8)
  grid_size =   (width/(block_size[0]-2)+1,
                height/(block_size[0]-2)+1,
                depth/(block_size[0]-2)+1)

  # # Get block/grid size for step 4.
  # block_size2 = (10,10,10)
  # grid_size2  = (width/(block_size2[0]-2)+1,
  #               height/(block_size2[0]-2)+1,
  #               depth/(block_size2[0]-2)+1)

  # Initialize variables.
  labeled       = np.zeros([height,width,depth]) 
  labeled       = np.float64(labeled)
  width         = np.int32(width)
  height        = np.int32(height)
  depth         = np.int32(depth)
  count         = np.int32([0])

  # Transfer labels asynchronously.
  labeled_d = gpu.to_gpu_async(labeled)
  counters_d = gpu.to_gpu_async(count)
  # mask_d = cu.np_to_array( mask, order='C' )
  # cu.bind_array_to_texref(mask_d, mask_texture)
  # Bind CUDA textures.
  #I_cu = cu.matrix_to_array(I, order='C')
  I_cu = cu.np_to_array(I, order='C')
  cu.bind_array_to_texref(I_cu, image_texture)

  # Step 1.
  descent_kernel(labeled_d, width, height, depth, 
    block=block_size, grid=grid_size)
  start_time = cu.Event()
  end_time = cu.Event()
  start_time.record()


  counters_d = gpu.to_gpu(np.int32([0]))
  #counters_d = gpu.to_gpu_async(np.int32([0]))
  old, new = -1, -2; it = 0
  while old != new:
    it +=1 
    old = new
    plateau_kernel(labeled_d, counters_d, width, height, depth,
    block=block_size, grid=grid_size)
    new = counters_d.get()[0]
  print 'plateau kernel', it-2

  # Step 2.
  increment_kernel(labeled_d,width,height,depth, 
    block=block_size,grid=grid_size)

  counters_d = gpu.to_gpu(np.int32([0]))
  old, new = -1, -2; it = 0

  while old != new:
    it +=1
    old = new
    minima_kernel(labeled_d, counters_d, width, height, depth,
    block=block_size, grid=grid_size)
    new = counters_d.get()[0]
  print 'minima kernel', it-2

  # Step 3.
  # counters_d = gpu.to_gpu(np.int32([0]))
  # old, new = -1, -2; it = 0
  # while old != new:
  #   it +=1
  #   old = new
  #   plateau_kernel(labeled_d, counters_d, width,
  #   height, depth, block=block_size, grid=grid_size)
  #   new = counters_d.get()[0]
  # print 'plateau kernel', it-2
  
  # Step 4
  counters_d = gpu.to_gpu(np.int32([0]))
  old, new = -1, -2; it = 0
  while old != new:
    it +=1
    old = new
    flood_kernel(labeled_d, counters_d, width,
    height, depth, block=block_size, grid=grid_size)
    new = counters_d.get()[0]
  print 'flood kernel', it-2

  labels = labeled_d.get()
  labels = labels*mask
  
  # End GPU timers.
  end_time.record()
  end_time.synchronize()
  gpu_time = start_time.\
  time_till(end_time) * 1e-3

  # print str(gpu_time)
  #cu.DeviceAllocation.free(counters_d)
  del counters_d

  return labels

if __name__ == '__main__':
  # Show the usage information.
  if len(argv) != 2:
    print "Usage: python ws_gpu.py test.dcm"
  # Read in the DICOM image data.
  #O = read_dcm(argv[1])
  # Preprocess the image.
  #I = preprocess(O)
  # Get the watershed transform.
  L = watershed(I)
  # Show the final edges.
  showEdges(L,O)
