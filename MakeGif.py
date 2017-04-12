import imageio as io
from IO_utils import *
files = find_files('./tmp', pattern="*.png")
#making animation
# writer = io.get_writer('smooth.gif', mode='I')
# print file_names
# for filename in file_names:
# 	image = io.imread(filename)
# 	#import IPython; IPython.embed()
# 	writer.append_data(image)
# writer.close()
images = [io.imread(file) for file in files]
io.mimsave('smooth.mp4', images)