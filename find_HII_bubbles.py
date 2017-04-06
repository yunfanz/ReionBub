from Choud14 import *
from tocmfastpy import *
from IO_utils import *
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cu
import pycuda.autoinit
from pyfft.cuda import Plan
from pycuda.tools import make_default_context

def find_bubbles(I, fil='kspace'):
	"""brute force method"""
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
	if fil == 'rspace':
		kernel = main_module.get_function("real_tophat_kernel")
	elif fil == 'kspace':
		kernel = main_module.get_function("k_tophat_kernel")
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
	ionized_d = gpuarray.to_gpu_async(ionized)
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
		kernel(ionized_d, width, R, S0, block=block_size, grid=grid_size)
		end.record()
		end.synchronize()
		R *= (1./1.5)

	ionized = ionized_d.get()
	return ionized

def conv(delta_d, filt_d, shape, fil):
	smoothI = np.zeros(shape, dtype=np.complex64)
	smoothed_d = gpuarray.to_gpu(smoothI)
	plan = Plan(shape, dtype=np.complex64)
	plan.execute(delta_d)
	if fil == 'rspace':
		plan.execute(filt_d)
	smoothed_d = delta_d * filt_d.conj()
	plan.execute(smoothed_d, inverse=True)
	return smoothed_d.real

def conv_bubbles(I, fil='kspace'):
	"""uses fft convolution"""
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
	update_kernel = main_module.get_function("update_kernel")
	real_tophat = main_module.get_function("real_tophat")
	k_tophat = main_module.get_function("k_tophat")
	image_texture = main_module.get_texref("img")

	# Get contiguous image + shape.
	height, width, depth = I.shape
	
	
	# Get block/grid size for steps 1-3.
	block_size =  (8,8,8)
	grid_size =   (width/(block_size[0]-2)+1,
				height/(block_size[0]-2)+1,
				depth/(block_size[0]-2)+1)
	 # Initialize variables.
	ionized       = np.zeros([height,width,depth]) 
	ionized       = np.float32(ionized)
	width         = np.int32(width)
	I             = np.float32(I.copy()*fgrowth)
	smoothI       = np.zeros(I.shape, dtype=np.complex64)
	filt          = np.ones_like(I)


	# Transfer labels asynchronously.
	ionized_d = gpuarray.to_gpu_async(ionized)
	delta_d = gpuarray.to_gpu_async(I)
	# I_cu = cu.np_to_array(I, order='C')
	# cu.bind_array_to_texref(I_cu, image_texture)

	plan = Plan(I.shape, dtype=np.complex64)
	R = RMAX
	while R > RMIN:
		print R
		R = np.float32(R)

		S0 = np.float32(sig0(R))

		start = cu.Event()
		step1 = cu.Event()
		step2 = cu.Event()
		end = cu.Event()

		start.record()
		filt_d = gpuarray.to_gpu(filt)
		if fil == 'kspace':
			ks = (9*np.pi/2)**(1./3)/R
			k_tophat(filt_d, width, ks, block=block_size, grid=grid_size)
		elif fil == 'rspace':
			real_tophat(filt_d, width, R, block=block_size, grid=grid_size)
		step1.record(); step1.synchronize()
		smoothed_d = conv(delta_d.astype(np.complex64), filt_d.astype(np.complex64), I.shape, fil=fil)
		import IPython; IPython.embed()
		step2.record(); step2.synchronize()
		update_kernel(ionized_d, smoothed_d, width, R, S0, block=block_size, grid=grid_size)
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
	ion_field = find_bubbles(d1, fil='rspace')
	print ion_field.shape
	import IPython; IPython.embed()
