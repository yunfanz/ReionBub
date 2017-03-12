import itertools
import numpy as n
import matplotlib.pyplot as plt
import cosmolopy.perturbation as pb
import cosmolopy.density as cd
from scipy.integrate import quad
from scipy.interpolate import interp1d, interp2d
cosmo = {'baryonic_effects':True,'omega_k_0':0,'omega_M_0':0.315, 'omega_b_0':0.0487, 'n':0.96, 'N_nu':0, 'omega_lambda_0':0.685,'omega_n_0':0., 'sigma_8':0.829,'h':0.673}


def polyval2d(x, y, m):
    order = int(n.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = n.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z
def RG(RL): return 0.46*RL
def W(y): return 3/y**3*(n.sin(y)-y*n.cos(y))
def WG(y): return n.exp(-y**2/2)
def Del2k(k):
    Pk = pb.power_spectrum(k,0.,**cosmo)
    Del2k = k**3*Pk/2/n.pi**2
    #fgrowth = pb.fgrowth(z, cosmo['omega_M_0']) 
    #Del2k0 = Del2k/fgrowth**2#*pb.norm_power(**cosmo)
    return Del2k
def sig1m(RL,kf=10.):
	kmax = kf/RL
	return quad(lambda k: Del2k(k)*W(RL*k)*WG(RG(RL)*k)/k, 0, kmax)[0]
dsig1m = n.load('sig1m.npz')
sig1mRl,sig1marr = dsig1m['arr_0'],dsig1m['arr_1']
sig1mint = interp1d(sig1mRl,sig1marr,kind='cubic')
def sig1ms(RL,kmin=0.01,kmax=20.,Nk=100):
	logk = n.linspace(n.log(kmin),n.log(kmax),Nk)
	k = n.exp(logk)
	return n.sum(Del2k(k)*k**2*n.exp(-k**2*RG(RL)**2/2)*W(RL*k))*(logk[1]-logk[0])
def sig1mfit(RL,kmax=20.):
	#coeff = n.array([-2414.54936415,  1687.14927674,  -439.55170349,    62.80654235])
	coeff = n.array([-35197.22457096,  44816.6140037 , -22450.21477783,   5671.79478317,
         -790.99091133,     74.00855598])
	return n.poly1d(coeff)(RL)

def SX(RL,R0,kf=10.): 
    kmax = kf/R0
    return quad(lambda k: Del2k(k)*W(RL*k)*W(R0*k)/k, 0, kmax)[0]
def sig1mX(RL,R0,kf=10.):
    kmax = kf/R0
    return quad(lambda k: Del2k(k)*(k**2)*WG(RG(RL)*k)*W(R0*k)/k, 0, kmax)[0]
def SXlog(RL,R0,kf=10.): 
    logmax = n.log(kf/R0)
    return quad(lambda logk: Del2k(n.exp(logk))*W(RL*n.exp(logk))*W(R0*n.exp(logk)), -1000., logmax)[0]
def sig1mXlog(RL,R0,kf=10.):
    logmax = n.log(kf/R0)
    return quad(lambda logk: Del2k(n.exp(logk))*(n.exp(logk)**2)*WG(RG(RL)*n.exp(logk))*W(R0*n.exp(logk)), -1000., logmax)[0]
def sig1mXlogs(RL,R0,kf=10.):
    logmax = n.log(kf/R0)
    logk = n.arange(-100,logmax,1)
    return sum(Del2k(n.exp(logk))*(n.exp(logk)**2)*WG(RG(RL)*n.exp(logk))*W(R0*n.exp(logk)))*(logk[1]-logk[0])
def sig1mXfit(RL,R0):
	logr,logR0 = n.log(RL),n.log(R0)

	coeff = n.array([ -28.34342859,   97.1097302 ,  -72.45264961,   14.71510979,
       -161.41422298,  323.64514131, -191.69521191,   34.23932032,
       -147.24355272,  167.14788951,  -48.49505801,    2.58502259,
        -31.62817142,   28.31661388,   -4.88490162,   -0.33095606])
	return polyval2d(logr,logR0,coeff)
