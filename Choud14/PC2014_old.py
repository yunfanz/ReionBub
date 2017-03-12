import numpy as n, matplotlib.pyplot as p, scipy.special
import cosmolopy.perturbation as pb
import cosmolopy.density as cd
from scipy.integrate import quad, tplquad
import itertools
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS

Om,sig8,ns,h,Ob = 0.315, 0.829, 0.96, 0.673, 0.0487
cosmo = {'baryonic_effects':True,'omega_k_0':0,'omega_M_0':0.315, 'omega_b_0':0.0487, 'n':0.96, 'N_nu':0, 'omega_lambda_0':0.685,'omega_n_0':0., 'sigma_8':0.829,'h':0.673}

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
def RG(RL): return 0.46*RL
def W(y): return 3/y**3*(n.sin(y)-y*n.cos(y))
def WG(y): return n.exp(-y**2/2)
def Del2k(k):
	Pk = pb.power_spectrum(k,0.,**cosmo)
	Del2k = k**3*Pk/2/n.pi**2
	#fgrowth = pb.fgrowth(z, cosmo['omega_M_0']) 
	#Del2k0 = Del2k/fgrowth**2#*pb.norm_power(**cosmo)
	return Del2k
#def sig0(RL,Del2k):
#	return n.sum(Del2k**2*W(RL*k)**2)*(logk[1]-logk[0])
#def sig0(RL,Del2k):
#	return n.sum(Del2k**2*W(RL*k)**2/k)*(k[1]-k[0])
def polyval2d(x, y, m):
    order = int(n.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = n.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z
def sig0test(RL,kmax):
	return quad(lambda k: Del2k(k)*W(RL*k)**2/k, 0, kmax)[0]   #z=0 extrapolated to present
def sig0(RL):
	return (pb.sigma_r(RL,0.,**cosmo)[0])**2
def sigG(RL,j): 
	return (pb.sigma_j(RL,j,0.,**cosmo)[0])**2
dsig1m = n.load('sig1m.npz')
sig1mRl,sig1marr = dsig1m['arr_0'],dsig1m['arr_1']
fs1m = interp1d(sig1mRl,sig1marr,kind='cubic')
def sig1m(RL):
	return fs1m(RL)
#def sig1m(RL,kmax=20.):
	#coeff = n.array([-35197.22457096,  44816.6140037 , -22450.21477783,   5671.79478317,
     #    -790.99091133,     74.00855598])
	#n.array([  1.81095565,  -6.51689501,   0.03932317,  12.22205831])
	#return n.poly1d(coeff)(RL)
	#return quad(lambda k: Del2k(k)*W(RL*k)*WG(RG(RL)*k)/k, 0, kmax)[0]
	#return n.sum(Del2k*k**2*n.exp(-k**2*RG(RL)**2/2)*W(RL*k))*(logk[1]-logk[0])
#def SX(RL,R0,kmax=20.): 
	#coeff = n.array([22.25,-6.645,0.54936,0.0128,18.66,6.029,-0.4879,0.01109,4.8616,-1.4594,0.1096,-0.00235,-0.384,0.107,-0.00741,0.0])
	#return polyval2d(RL,R0,coeff)
	#return quad(lambda k: Del2k(k)*W(RL*k)*W(R0*k)/k, 0, kmax)[0]
#def sig1mX(RL,R0,kmax=20.):
	#logr,logR0 = n.log(RL),n.log(R0)

	#coeff = n.array([   7.08046191,   28.16149525,  -23.50798007,    4.20273492,
    #    -34.31345153,  101.96878325,  -78.59663353,   16.35608005,
    #    -35.10071616,    1.19563953,   18.76803373,   -5.08233304,
    #     -7.29945622,   -5.95674768,    9.93434604,   -2.36906904])
	#return polyval2d(logr,logR0,coeff)
	#return quad(lambda k: Del2k(k)*(k**2)*WG(RG(RL)*k)*W(R0*k)/k, 0, kmax)[0]
#def SX(RL,R0,kf=20.): 
#    kmax = kf/R0
#    return quad(lambda k: Del2k(k)*W(RL*k)*W(R0*k)/k, 0, kmax)[0]
#def sig1mX(RL,R0,kf=20.):
#    kmax = kf/R0
#    return quad(lambda k: Del2k(k)*(k**2)*WG(RG(RL)*k)*W(R0*k)/k, 0, kmax)[0]
dSX = n.load('logSX.npz')
lSXRl,lSXR0,arrSX = dSX['arr_0'],dSX['arr_1'],dSX['arr_2']
fSX = RBS(lSXRl,lSXR0,arrSX)
def SX(RL,R0):
	res = fSX(n.log(RL),n.log(R0))
	if res.size > 1: print 'Warning: SX called with array instead of single number'
	return res[0][0]
ds1mX = n.load('logsig1mX.npz')
ls1mXRl,ls1mXR0,arrs1mX = ds1mX['arr_0'],ds1mX['arr_1'],ds1mX['arr_2']
fs1mX = RBS(ls1mXRl,ls1mXR0,arrs1mX)
def sig1mX(RL,R0):
	res = fs1mX(n.log(RL),n.log(R0))
	if res.size > 1: print 'Warning: s1mX called with array instead of single number'
	return res[0][0]
#def SX(RL,R0,kf=10.): 
#    logmax = n.log(kf/R0)
#    return quad(lambda logk: Del2k(n.exp(logk))*W(RL*n.exp(logk))*W(R0*n.exp(logk)), 0, logmax)[0]
#def sig1mX(RL,R0,kf=10.):
#    logmax = n.log(kf/R0)
#    return quad(lambda logk: Del2k(n.exp(logk))*(n.exp(logk)**2)*WG(RG(RL)*n.exp(logk))*W(R0*n.exp(logk)), 0, logmax)[0]
def gam(RL):
	return sig1m(RL)/n.sqrt(sig0(RL)*sigG(RL,2))
def Vstar(RL):
	return (6*n.pi)**1.5*(sigG(RL,1)/sigG(RL,2))**3
def erf(x):
	return scipy.special.erf(x)
def prob(x,av=0.5,var=0.25):
	return 1/n.sqrt(2*n.pi*var)/x*n.exp(-(n.log(x)-av)**2/2/var)
def F(x):
	return (x**3-3*x)/2*(erf(x*n.sqrt(5./2))+erf(x*n.sqrt(5./8)))+n.sqrt(2./5/n.pi)*((31*x**2/4+8./5)*n.exp(-5.*x**2/8)+(x**2/2-8./5)*n.exp(-5.*x**2/2))
def Deltac(z):
	fgrowth = pb.fgrowth(z, cosmo['omega_M_0'])    # = D(z)/D(0)
	return 1.686/fgrowth
def pG(y,av,var):
	return 1/n.sqrt(2*n.pi*var)*n.exp(-(y-av)**2/2/var)
def B(z,beta,s):
	return Deltac(z)+beta*n.sqrt(s)
def Q(m,M0):
	r,R0 = m2R(m), m2R(M0)
	s,s0 = sig0(r), sig0(R0)
	sx = SX(r,R0)
	return 1-sx**2/s/s0
def epX(m,M0):
	r,R0 = m2R(m), m2R(M0)
	s,s0 = sig0(r), sig0(R0)
	sx = SX(r,R0)
	sg1m = sig1m(r)
	sg1mX = sig1mX(r,R0)
	return s*sg1m/sx/sg1mX
def subgrand(b,del0,m,M0,z):
	V,r,dmdr = pb.volume_radius_dmdr(m,**cosmo)
	R0 = m2R(M0)
	s,s0,sx = sig0(r), sig0(R0),SX(r,R0)
	Bb = B(z,b,s)
	gamm = gam(r)
	epx,q = epX(m,M0), Q(m,M0)
	print 'gamm,epx,q =',gamm,epx,q 
	meanmu = del0/n.sqrt(s)*sx/s0
	varmu = Q(m,M0)
	meanx = gamm*((Bb-del0*sx/s0)*(1-epx)/q/n.sqrt(s)+Bb*epx/n.sqrt(s))
	varx = 1-gamm**2-gamm**2*(1-epx)**2*(1-q)/q 
	fact = V/Vstar(R0)*pG(Bb/n.sqrt(s),meanmu, varmu)
	print b, Bb/n.sqrt(s),meanmu,varmu,pG(Bb/n.sqrt(s),meanmu, varmu)
	factint = quad(lambda x: (x/gamm-b)*F(x)*pG(x,meanx,varx),b*gamm,100)[0]
	#print fact, factint
	return fact*factint
def integrand(del0,m,M0,z):  #2s*f_ESP
	s = sig0(m2R(m))
	print '#################'
	return quad(lambda b: prob(b)*subgrand(b,del0,m,M0,z),0,4.)[0]/2/s
def dsdm(m):
	return (sig0(m2R(m+1))-sig0(m2R(m-1)))/2
def fcoll(del0,M0,z):
	return quad(lambda m: integrand(del0,m,M0,z)*dsdm(m),mmin(z),M0)

def All(x,b,m,del0,M0,z): #z,y,x,c,c,c
	V,r,dmdr = pb.volume_radius_dmdr(m,**cosmo)
	R0 = m2R(M0)
	s,s0,sx = sig0(r), sig0(R0),SX(r,R0)
	Bb = B(z,b,s)
	gamm = gam(r)
	epx,q = epX(m,M0), Q(m,M0)
	#print 'gamm,epx,q =',gamm,epx,q 
	meanmu = del0/n.sqrt(s)*sx/s0
	varmu = Q(m,M0)
	meanx = gamm*((Bb-del0*sx/s0)*(1-epx)/q/n.sqrt(s)+Bb*epx/n.sqrt(s))
	varx = 1-gamm**2-gamm**2*(1-epx)**2*(1-q)/q 
	fact = V/Vstar(R0)*pG(Bb/n.sqrt(s),meanmu, varmu)
	#print b, Bb/n.sqrt(s),meanmu,varmu,pG(Bb/n.sqrt(s),meanmu, varmu)
	return fact*prob(b)*(x/gamm-b)*F(x)*pG(x,meanx,varx)/2/sig0(m2R(m))*dsdm(m)

p.figure()
Z = [12.]
###################### PARAMETERS ############################
#z = 12.
for z in Z:
	deltac = Deltac(z)
	#deltac = 1.686*(1+z)    #z_eq =3233?
	##print deltac
	#Del2k0 = Del2k/fgrowth**2     #linearly extrapolated to present epoch
	####################################
	#sig_8 = n.sqrt(sig0(8./cosmo['h'],Del2k0))
	#print sig_8
	sig_8 = n.sqrt(sig0(8./cosmo['h']))
	print 'sig_8',sig_8
	#Del2k0 = Del2k0*(sig8/sig_8)
	####################################
	zeta = 40.
	K = scipy.special.erfinv(1-1./zeta)
	print 'K(zeta)=',K
	#import IPython; IPython.embed()
	####################### FZH04 ##############################
	##### m_min
	Tvir = 1.E4
	#mmin = (Tvir/442/Om**(1./3)/((1+z)/100))**(3./2)*(h**(-1)*1.E4)
	mmin = pb.virial_mass(Tvir,z,**cosmo)
	print "minimum mass (msuns)", mmin
	RLmin = m2R(mmin)
	print 'R',RLmin
	#rlmin = pb.mass_to_radius(mmin,**cosmo)
	#print RLmin, rlmin #==
	#smin = sig0(RLmin,Del2k0)
	smin = sig0(RLmin)
	print 'smin=',smin
	#######
	S0max = sig0(m2R(zeta*mmin))
	S0 = n.arange(0,S0max,0.2)
	BFZH = deltac-n.sqrt(2*(smin-S0))*K
	bFZH0 = deltac-K*n.sqrt(2*smin)
	bFZH1 = K/n.sqrt(2*smin)
	BFZHlin = bFZH0+bFZH1*S0

	p.plot(S0,BFZH,'b')
	p.plot(S0,BFZHlin,'b.-')


	M0 = zeta*mmin*2
	del0 = 5.
	#print quad(lambda m: integrand(del0,m,M0,12.)*dsdm(m),mmin,M0)
	tplquad(All,mmin,M0,lambda x: 0, lambda x: 5., lambda x,y: gam(m2R(x))*y,lambda x,y: 10.,args=(del0,M0,z))
p.show()

