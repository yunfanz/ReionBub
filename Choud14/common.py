from sigmas import *
import sigmas
COMMONDIR = os.path.dirname(sigmas.__file__)
print COMMONDIR
rhobar = cd.cosmo_densities(**cosmo)[1]  #msun/Mpc

def m2R(m):
	RL = (3*m/4/np.pi/rhobar)**(1./3)
	return RL
def m2V(m):
	return m/rhobar
def R2m(RL):
	m = 4*np.pi/3*rhobar*RL**3
	return m

dmS = np.load(COMMONDIR+'/sig0.npz')
RLtemp, MLtemp,SLtemp = dmS['radius'], dmS['mass'],dmS['sig0']
fs2m = interp1d(SLtemp,MLtemp)
# fsig0 = interp1d(RLtemp,SLtemp)
# def sig0(RL):
# 	return fsig0(RL)
print 'generated fs2m'
def S2M(S):
	return fs2m(S)
def m2S(m):
	return sig0(m2R(m))
def mmin(z,Tvir=1.E4):
	return pb.virial_mass(Tvir,z,**cosmo)
def Deltac(z):
	fgrowth = pb.fgrowth(z, cosmo['omega_M_0'])    # = D(z)/D(0)
	return 1.686/fgrowth
	#return 1.686*fgrowth  
def fcoll_FZH(del0,M0,z,debug=False):
	# Eq. (6)
	print del0
	mm = mmin(z)
	R0 = m2R(M0)
	smin, S0 = sig0(m2R(mm)), sig0(R0)
	return erfc((Deltac(z)-del0)/np.sqrt(2*(smin-S0)))