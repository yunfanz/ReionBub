#Module of an older version of mul1 functions
import numpy as n, matplotlib.pyplot as p, scipy.special
import cosmolopy.perturbation as pb
import cosmolopy.density as cd
from scipy.integrate import quad, tplquad
import itertools
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
import optparse, sys
import gen_inter as GI
# o = optparse.OptionParser()
# o.add_option('-d','--del0', dest='del0', default=5.)
# o.add_option('-m','--mul', dest='mul', default=1.)
# o.add_option('-z','--red', dest='red', default=12.)
# opts,args = o.parse_args(sys.argv[1:])
# print opts, args

Om,sig8,ns,h,Ob = 0.315, 0.829, 0.96, 0.673, 0.0487
cosmo = {'baryonic_effects':True,'omega_k_0':0,'omega_M_0':0.315, 'omega_b_0':0.0487, 'n':0.96, 'N_nu':0, 'omega_lambda_0':0.685,'omega_n_0':0., 'sigma_8':0.829,'h':0.673}
rhobar = cd.cosmo_densities(**cosmo)[1]
def m2R(m):
	  #msun/Mpc
	RL = (3*m/4/n.pi/rhobar)**(1./3)
	return RL
def m2V(m):
	return m/rhobar

def R2m(RL):
	m = 4*n.pi/3*rhobar*RL**3
	return m
def mmin(z,Tvir=1.E4):
	return pb.virial_mass(Tvir,z,**cosmo)
m12 = mmin(12.)
r12 = m2R(m12)

