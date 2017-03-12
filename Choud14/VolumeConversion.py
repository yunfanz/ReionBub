import pylab as p
import numpy as n
import cosmolopy.perturbation as pb
import cosmolopy.density as cd
from scipy.integrate import quad
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import RectBivariateSpline as RBS



def del0over1plusz(th):
	return 3./20*(6*(th-n.sin(th)))*(2./3)
def redshift(del0,th):
	RHS = del0over1plusz(th)/del0
	return 1/RHS-1

del0 = 3.
theta = n.arange(1.2,4.,0.01)
#Z = redshift(del0,theta)
d0over1plusz = del0over1plusz(theta)
