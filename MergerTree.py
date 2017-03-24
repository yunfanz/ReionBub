import numpy as np
from IO_utils import *
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage
from skimage import measure, morphology, segmentation
from ete3 import Tree, TreeStyle, NodeStyle, faces, AttrFace, CircleFace


class Bubble(measure._regionprops._RegionProperties):
	def centroid(self):
		return tuple(self.coords.mean(axis=0))

def bubbleprops(label_image, intensity_image=None, cache=True, areasort=True):
    label_image = np.squeeze(label_image)
    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2-D and 3-D images supported.')
    if not np.issubdtype(label_image.dtype, np.integer):
        raise TypeError('Label image must be of integral type.')
    bubbles = []
    objects = ndimage.find_objects(label_image)
    for i, sl in enumerate(objects):
        if sl is None:
            continue
        label = i + 1
        props = Bubble(sl, label, label_image, intensity_image, cache)
        bubbles.append(props)
    if areasort:
    	bubbles = sorted(bubbles, key=lambda r: r.area, reverse=True)
    return bubbles


def bbox_merit(b1L, b2L, b1R, b2R):
	bb = np.where(b1R<b2R, b1R, b2R) - np.where(b1L>b2L, b1L, b2L)
	vol1 = np.product(b1R-b1L)
	vol2 = np.product(b2R-b2L)
	if True:
		return float(np.product(bb))/max(vol1, vol2)
	else:
		return float(np.product(bb))**2/np.product(b1R-b1L)/np.product(b2R-b2L)

def get_merit(coords1, coords2):
	N1, N2 = coords1.shape[0], coords2.shape[0]
	N12 = float(len( set([tuple(row) for row in coords1]) & set([tuple(row) for row in coords2])))
	if True:
		#print 'N12, N1, N2', N12, N1, N2, N12/max(N1, N2)
		return N12/max(N1, N2)
	else:
		return N12**2/N1/N2

def get_bubbles(file, connectivity=3):
	img = np.load(file)['labels']
	labeled = measure.label(img, connectivity=connectivity)
	#mask = labeled > 0
	#labeled  = labeled *morphology.remove_small_objects(mask, 30)

	return bubbleprops(labeled)

class MergerTree:
	def __init__(self, files, maxnodes=1):
		self.Tree = Tree()
		self.TreeDepth = len(files)
		self.maxnodes = maxnodes
		self.files = files
		print files

	def build(self):
		Rup = None
		for n in xrange(self.TreeDepth):
			file = self.files[n]
			if n == 0:
				R = get_bubbles(file)
				R = [R[0]]  #only one root
				for r in R:
					name = str(n)+'_'+str(r.label)
					print name, r.area, r.centroid()
					self.Tree.add_child(name=name)
				Rup = R
			else:
				if len(Rup) == 0: break
				R = get_bubbles(file)
				newRup = []
				print "L", len(Rup), len(self.Tree.get_leaves())
				root = self.Tree.get_tree_root()
				for i, leaf in enumerate(self.Tree.get_leaves()):
					
					level = int(leaf.name.split('_')[0])
					#assert level == int(root.get_distance(leaf)-1)
					#print level, n, int(root.get_distance(leaf)-1)
					if level != n-1: continue  #make sure we're operating on the lower level leaves only 
					print leaf
					ind = int(leaf.name.split('_')[1])
					rup = None
					for r in Rup:
						if r.label == ind:
							rup = r
					if rup is None:
						print ind, "not found!"
						continue
					leaf.add_features(vol=rup.area)
					
					assert ind == int(rup.label)
					# R, merits = self.find_merger(rup, R, returnmerit=True)
					# print merits
					R = self.find_merger(rup, R)
					for r in R:
						name = str(n)+'_'+str(r.label)
						print name
						leaf.add_child(name=name)
					newRup += R
				Rup = newRup

	def find_merger(self, r1, R2, returnmerit=False):
		print len(R2)
		R2 = self.overlap_bbox(r1, R2)
		print len(R2)
		R2 = sorted(R2, key=lambda r: get_merit(r1.coords, r.coords), reverse=True)
		if len(R2)> self.maxnodes:
			R2 = R2[:self.maxnodes]
		if returnmerit:
			merits = []
			for r2 in R2:
				merits.append(get_merit(r1.coords, r2.coords))
			return R2, merits
		return R2

	def overlap_bbox(self, r1, R2, keep=20):
		"""keep the top keep number of regions with overlapping bbox merit"""
		b1L = np.asarray(r1.bbox[:3])  #lower bounds
		b1R = np.asarray(r1.bbox[3:])  #upper bounds
		realR2 = []; BMerit = []
		for r2 in R2:
			b2L = np.asarray(r2.bbox[:3])
			b2R = np.asarray(r2.bbox[3:])

			if ( (b1L<b2R).all() and (b1R>b2L).all() ):
				BMerit.append(bbox_merit(b1L, b2L, b1R, b2R))
				realR2.append(r2)

		keepind = [ind for (merit,ind) in sorted(zip(BMerit,range(len(BMerit))),reverse=True)]
		if len(keepind)> keep:
			keepind = keepind[:keep]

		return [realR2[ind] for ind in keepind]




def layout(node):
    if node.is_leaf():
        # Add node name to laef nodes
        N = AttrFace("name", fsize=14, fgcolor="black")
        faces.add_face_to_node(N, node, 0)
    if "vol" in node.features:
        # Creates a sphere face whose size is proportional to node's
        # feature "weight"
        print node.vol
        C = CircleFace(radius=float(node.vol)/5000, color="RoyalBlue", style="sphere")
        # Let's make the sphere transparent
        C.opacity = 0.3
        # And place as a float face over the tree
        faces.add_face_to_node(C, node, 0, position="float")

def get_style():
	# Create an empty TreeStyle
    ts = TreeStyle()
    # Set our custom layout function
    ts.layout_fn = layout
    # Draw a tree
    ts.mode = "c"
    # We will add node names manually
    ts.show_leaf_name = True
    # Show branch data
    ts.show_branch_length = True
    ts.show_branch_support = True
    return ts
def parse_name(name):
	return [int(num) for num in name.split('_')]

def get_top_dict(T):
	cnt = 0
	dic = {}
	for node in T.Tree.traverse('preorder'):
		if cnt >T.TreeDepth: break
		cnt += 1; print node.name
		for child in node.get_children():
			lvl, lab = parse_name(child.name)
			dic[lvl] = dic.get(lvl, []) + [lab]
	return dic



if __name__=='__main__':
	#DIR = '/data2/lin0_logz10-15_zeta40/Boxes/'
	#DIR = '/home/yunfanz/Data/21cmFast/Boxes/'
	DIR = './NPZ/'
	files = find_files(DIR)
	T = MergerTree(files, maxnodes=3)
	T.build()
	#import IPython; IPython.embed()
	ts = get_style()
	print T.Tree
	dic = get_top_dict(T)
	print dic
	save_obj(dic, '3nodeTree')
	#T.Tree.show(tree_style=ts)
	import IPython; IPython.embed()
