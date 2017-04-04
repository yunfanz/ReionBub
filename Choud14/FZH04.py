import numpy as n, matplotlib.pyplot as p, scipy.special
import cosmolopy.perturbation as pb
import cosmolopy.density as cd
from scipy.integrate import quad,tplquad
import itertools
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
import optparse, sys
from sigmas import sig0
o = optparse.OptionParser()
o.add_option('-d','--del0', dest='del0', default=5.)
o.add_option('-m','--mul', dest='mul', default=1.)
o.add_option('-z','--red', dest='red', default=12.)
opts,args = o.parse_args(sys.argv[1:])
print opts, args

Om,sig8,ns,h,Ob = 0.315, 0.829, 0.96, 0.673, 0.0487
Planck13 = {'baryonic_effects':True,'omega_k_0':0,'omega_M_0':0.315, 'omega_b_0':0.0487, 'n':0.96, 'N_nu':0, 'omega_lambda_0':0.685,'omega_n_0':0., 'sigma_8':0.829,'h':0.673}
cosmo = Planck13
def m2R(m):
	rhobar = cd.cosmo_densities(**cosmo)[1]  #msun/Mpc
	RL = (3*m/4/n.pi/rhobar)**(1./3)
	return RL
def m2V(m):
	rhobar = cd.cosmo_densities(**cosmo)[1]  #msun/Mpc
	return m/rhobar

def R2m(RL):
	rhobar = cd.cosmo_densities(**cosmo)[1]  #msun/Mpc
	m = 4*n.pi/3*rhobar*RL**3
	return m
def mmin(z,Tvir=1.E4):
	return pb.virial_mass(Tvir,z,**cosmo)
dmS = n.load('m2S.npz')
MLtemp,SLtemp = dmS['arr_0'],dmS['arr_1']
fs2m = interp1d(SLtemp,MLtemp,kind='cubic')
def S2M(S):
	return fs2m(S)
def Deltac(z):
	fgrowth = pb.fgrowth(z, cosmo['omega_M_0'])    # = D(z)/D(0)
	return 1.686/fgrowth
	#return 1.686*fgrowth  

######################## SIZE DISTRIBUTION #############################
	####################### FZH04 ##############################

def fFZH(S,zeta,B0,B1):
	res = B0/n.sqrt(2*n.pi*S**3)*n.exp(-B0**2/2/S-B0*B1-B1**2*S/2)
	return res
def BFZH(S0,deltac,smin,K):
	return deltac-n.sqrt(2*(smin-S0))*K
def BFZHlin(S0,deltac,smin,K):
	b0 = deltac-K*n.sqrt(2*smin)
	b1 = K/n.sqrt(2*smin)
	return b0+b1*S0
def dlnBFdlnS0(S0,deltac,smin,K,d=0.001):
	Bp,Bo,Bm = BFZH(S0+d,deltac,smin,K), BFZH(S0,deltac,smin,K), BFZH(S0-d,deltac,smin,K)
	return S0/Bo*(Bp-Bm)/2/d
def dlnBFlindlnS0(S0,deltac,smin,K,d=0.001):
	Bp,Bo,Bm = BFZHlin(S0+d,deltac,smin,K), BFZHlin(S0,deltac,smin,K), BFZHlin(S0-d,deltac,smin,K)
	return S0/Bo*(Bp-Bm)/2/d

	##### m_min
dDoZ = n.load('theta.npz')
thetal,DoZl = dDoZ['arr_0'],dDoZ['arr_1']
ftheta = interp1d(DoZl,thetal,kind='cubic')
def theta(z,del0):
	return ftheta(del0/(1+z))
def RphysoR0(del0,z):
	th = theta(z,del0)
	return 3./10/del0*(1-n.cos(th))
def RcovEul(del0,z):
	return RphysoR0(del0,z)*(1+z)
def dlinSdlnR(lnR,d=0.001):
	res = (n.log(sig0(n.exp(lnR+d)))-n.log(sig0(n.exp(lnR-d))))/d/2
	return n.abs(res)




################################## MAIN ######################################




for z in [12., 16.]:
	PLOT = True
	zeta = 40.
	K = scipy.special.erfinv(1-1./zeta)
	Tvir = 1.E4
	#z = 12.
	deltac = Deltac(z)
	mm = mmin(z)
	M0min = zeta*mm
	RLmin,R0min = m2R(mm), m2R(M0min)
	print 'R',RLmin
	smin = sig0(RLmin)
	Rmin = R0min*RcovEul(deltac,z)  #S0=smin, so del0=deltac; convertion from lagragian to comoving eulerian 
	####### FZH04 #######
	bFZH0 = deltac-K*n.sqrt(2*smin)
	bFZH1 = K/n.sqrt(2*smin)
	#bFZH = deltac-n.sqrt(2*(smin-S0))*K
	#bFZHlin = bFZH0+bFZH1*S0
	def dlnRdlnR0(lnR0,S0,del0):
		S0 = sig0(n.exp(lnR0))
		del0 = BFZH(S0,deltac,smin,K)
		th = theta(z,del0)
		thfactor = 1-3./2*th*(th-n.sin(th))/(1-n.cos(th))**2
		res = 1-dlinSdlnR(lnR0)*dlnBFdlnS0(S0,deltac,smin,K)*thfactor
		return res
	def V0dndlnR0(lnR0):
		S0 = sig0(n.exp(lnR0))
		return S0*fFZH(S0,zeta,bFZH0,bFZH1)*dlinSdlnR(lnR0)
	def VdndlnR0(lnR0):
		S0 = sig0(n.exp(lnR0))
		del0 = BFZHlin(S0,deltac,smin,K)
		#lnR0 = n.log(n.exp(lnR)/RcovEul(del0,z))
		
		VoV0 = (RcovEul(del0,z))**3
		#return VoV0/dlnRdlnR0(lnR0,S0,del0)*S0*fFZH(S0,zeta,bFZH0,bFZH1)*dlinSdlnR(lnR0)
		return VoV0*S0*fFZH(S0,zeta,bFZH0,bFZH1)*dlinSdlnR(lnR0)
	def VdndlnR(lnR0):
		S0 = sig0(n.exp(lnR0))
		del0 = BFZH(S0,deltac,smin,K)
		VoV0 = (RcovEul(del0,z))**3
		return VoV0/dlnRdlnR0(lnR0,S0,del0)*S0*fFZH(S0,zeta,bFZH0,bFZH1)*dlinSdlnR(lnR0)
	if True:
		print 'computing z=',z
		#Q = quad(lambda lnR: VdndlnR(lnR),n.log(Rmin),3.5)  #integrated over eulerian coordinates
		Q = quad(lambda lnR0: VdndlnR0(lnR0),n.log(R0min),3.5)  #integrated over eulerian coordinates
		print 'Q=',Q
		Q = Q[0]

		#######
		lnR0 = n.arange(n.log(R0min),3,0.03)
		S0list = []
		for lnr0 in lnR0: S0list.append(sig0(n.exp(lnr0)))
		S0list = n.array(S0list)
		#lnR = n.arange(n.log(Rmin),3,0.1)
		del0list = BFZH(S0list,deltac,smin,K)
		lnR = n.log(n.exp(lnR0)*RcovEul(del0list,z))

		normsize = []
		for lnr0 in lnR0:
			res = VdndlnR(lnr0)/Q
			print n.exp(lnr0),res
			normsize.append(res)
		p.figure(1)
		p.semilogx(n.exp(lnR),normsize,label=str(z))
		p.legend()

	
	if True:
		S0max = sig0(m2R(M0min))
		S0 = n.arange(0,S0max,0.2)
		bFZH = deltac-n.sqrt(2*(smin-S0))*K
		bFZHlin = bFZH0+bFZH1*S0
		p.figure(2)
		p.plot(S0,bFZH,'b', label=str(z))
		p.plot(S0,bFZHlin,'b.-')
		p.ylim([0,20])
		p.xlim([0,25])
		p.legend()
	if False: #for benchmark
		for i in range(1000):
			S0max = sig0(m2R(M0min))
		S0 = n.arange(0,S0max,0.2)
		bFZH = deltac-n.sqrt(2*(smin-S0))*K
		bFZHlin = bFZH0+bFZH1*S0



	

p.show()

################
# Z = float(opts.red)
# M0 = zeta*mmin(Z)*float(opts.mul)
# del0 = float(opts.del0)
###########################

# dlist = n.linspace(8,10,10)
# for del0 in dlist:
# 	res = fcoll_trapz_log(del0,M0,Z)
# 	print m2S(M0), res[0]
# if False:
# 	p.figure()
# 	p.plot(res[1],res[2])
# 	p.show()
	#tplquad(All,mm,M0,lambda x: 0, lambda x: 5., lambda x,y: gam(m2R(x))*y,lambda x,y: 10.,args=(del0,M0,z))

