# ReionBub
This repository contains scripts used to study the statistics of reionization bubble sizes from the simulation 21cmFast ([[6]]). 
This projects requires the following external packages:

```
numpy, matplotlib, scipy(ndimage), scikit-image, pycuda, ete3, vispy, pandas, seaborn, pyqt4 etc.
```
## 3D H-minima Transform and Watershed with PyCUDA
A 3d watershed algorithm is included for segmentation. The watershed algorithm is partly based on [1-4] and modified from 5. In particular, a H-minima transform (also on Pycuda) is used to reduce over-segmentation.

Sample usage:
```
python watershed.py -d /directory/to/21cmFast/Boxes/
```

## MergerTree with ETE3
A python merger tree builder that scrolls through watershed results at different redshifts and build merger trees, storing various properties of the elements. 

## 3D rendering with Vispy:
```
render_box.py
```
This general script can render various boxes such as distance transform, ionization field, density etc. 
```
isosurface.py 
```
This script renders a single descendant bubble and its progenitors at each redshift slice, from the watershed and MergerTree output files. 

```VR_bubbles```
Renders 3D VR-goggle style movies. 
![result](animation.gif)

## IO
```tocmfastpy```  is a package that handles reading 21cmFast boxes, it is modified from [[7]]. 


**References**

[[1]](http://www.fem.unicamp.br/~labaki/Academic/cilamce2009/1820-1136-1-RV.pdf) Vitor B, Körbes A. Fast image segmentation by watershed transform on graphical hardware. In: Proceedings of the 17th International Conference on Systems, Signals and Image Processing, pp. 376-379, Rio de Janeiro, Brazil.

[[2]](http://www.lbd.dcc.ufmg.br/colecoes/wvc/2009/0012.pdf) Körbes A et al. 2009. A proposal for a parallel watershed transform algorithm for real-time segmentation. In: V Workshop de Visão Computacional, São Paulo, Brazil.

[[3]](http://parati.dca.fee.unicamp.br/media/Attachments/courseIA366F2S2010/aula10/ijncr.pdf) Körbes A et al. 2010. Analysis of a step-by-step watershed algorithm using CUDA. International Journal of Natural Computing Research. 1:16-28.

[[4]](http://parati.dca.fee.unicamp.br/media/Attachments/courseIA366F2S2010/aula10/ijncr.pdf) Körbes A et al. 2011. Advances on Watershed Processing on GPU Architectures. In: 10th International Symposium on Mathematical Morphology, Lake Maggiore, Italy.

[[5]](https://github.com/louismullie/watershed-cuda/blob/master/ws_gpu.py) ws_gpu.py by louismullie

[[6]](https://github.com/andreimesinger/21cmFAST) 21cmFast by Andrei Messinger

[[7]](https://github.com/pritchardjr/tocmfastpy) tocmfastpy by Jonathan Pritchard
