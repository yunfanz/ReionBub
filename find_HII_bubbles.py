from Choud14 import *
from tocmfastpy import *
from IO_utils import *
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit

def find_bubbles(I):
	zeta = 40.
	Z = 12.
	RMAX = 3.
	RMIN = 1.
	mm = mmin(Z)
	smin = sig0(m2R(mm))
	deltac = Deltac(Z)
	fgrowth = deltac/1.686
	"""find bubbbles for deltax box I"""
	kernel_source = open("find_bubbles.cu").read()
	kernel_code = kernel_source % {
        'DELTAC': deltac,
        'RMIN': RMIN,
        'SMIN': smin, 
        'ZETA': zeta
    }
	main_module = nvcc.SourceModule(kernel_code)
	real_tophat_kernel = main_module.get_function("real_tophat_kernel")
	image_texture = main_module.get_texref("img")

	# Get contiguous image + shape.
	height, width, depth = I.shape
	I = np.float32(I.copy()*fgrowth)

	# Get block/grid size for steps 1-3.
	block_size =  (8,8,8)
	grid_size =   (width/(block_size[0]-2)+1,
				height/(block_size[0]-2)+1,
				depth/(block_size[0]-2)+1)
	 # Initialize variables.
	ionized       = np.zeros([height,width,depth]) 
	ionized       = np.float32(ionized)
	width         = np.int32(width)

	# Transfer labels asynchronously.
	ionized_d = gpu.to_gpu_async(ionized)
	I_cu = cu.np_to_array(I, order='C')
	cu.bind_array_to_texref(I_cu, image_texture)

	
	R = RMAX
	while R > RMIN:
		print R
		R = np.float32(R)
		S0 = np.float32(sig0(R))
		start = cu.Event()
		end = cu.Event()
		start.record()
		real_tophat_kernel(ionized_d, width, R, S0, block=block_size, grid=grid_size)
		end.record()
		end.synchronize()
		R *= (1./1.1)

	ionized = ionized_d.get()
	return ionized

if __name__ == '__main__':
	o = optparse.OptionParser()
	o.add_option('-d','--dir', dest='DIR', default='/home/yunfanz/Data/21cmFast/Boxes/')
	(opts, args) = o.parse_args()

	z = 12.00
	files = find_deltax(opts.DIR, z=z)
	file = files[0]
	b1 = boxio.readbox(file)
	d1 = 1 - b1.box_data[::10, ::10, ::10]
	print d1.shape
	ion_field = find_bubbles(d1)
	print ion_field.shape
	import IPython; IPython.embed()
