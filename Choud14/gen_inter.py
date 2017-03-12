#Currently used to test and tune trapezoidal integration for the sigmas, used in mul1.py
import itertools
import numpy as n
import matplotlib.pyplot as plt
import cosmolopy.perturbation as pb
import cosmolopy.density as cd
from scipy.integrate import quad
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import RectBivariateSpline as RBS
cosmo = {'baryonic_effects':True,'omega_k_0':0,'omega_M_0':0.315, 'omega_b_0':0.0487, 'n':0.96, 'N_nu':0, 'omega_lambda_0':0.685,'omega_n_0':0., 'sigma_8':0.829,'h':0.673}


def RG(RL): return 0.46*RL
def W(y): return 3/y**3*(n.sin(y)-y*n.cos(y))
def WG(y): return n.exp(-y**2/2)
def Del2k(k):
    Pk = pb.power_spectrum(k,0.,**cosmo)
    Del2k = k**3*Pk/2/n.pi**2
    #fgrowth = pb.fgrowth(z, cosmo['omega_M_0']) 
    #Del2k0 = Del2k/fgrowth**2#*pb.norm_power(**cosmo)
    return Del2k
def sigint(r):
    kmax = 10./r
    return quad(lambda k: Del2k(k)*W(r*k)*WG(RG(r)*k)/k, 0.0001, kmax)[0]
#def sig1mlog(RL,R0,kf=10.): 
#    logmax = n.log(kf)
#    return quad(lambda logk: Del2k(n.exp(logk))*n.exp(logk)**2*WG(n.exp(logk)*RG(RL))*W(RL*n.exp(logk)), -100., logmax)[0]
def SXlog(RL,R0,kf=10.): 
    logmax = n.log(kf)
    return quad(lambda logk: Del2k(n.exp(logk))*W(RL*n.exp(logk))*W(R0*n.exp(logk)), -100., logmax)[0]
#def sig1mXlog(RL,R0,kf=10.): 
#    logmax = n.log(kf)
#    return quad(lambda logk: Del2k(n.exp(logk))*n.exp(logk)**2*WG(n.exp(logk)*RG(RL))*W(R0*n.exp(logk)), -100., logmax)[0]
def sigG(RL,j): 
    return (pb.sigma_j(RL,j,0.,**cosmo)[0])**2
def SX(RL,R0,kf=6.): 
    kmax = kf/RL
    return quad(lambda k: Del2k(k)*W(RL*k)*W(R0*k)/k, 0.001, kmax)
def sig1mX(RL,R0,kf=10.):
    kmax = kf/RL
    return quad(lambda k: Del2k(k)*(k**2)*WG(RG(RL)*k)*W(R0*k)/k, 0.001, kmax)
def sig1m(RL,R0,kf=10.):
    kmax = kf/RL
    return quad(lambda k: Del2k(k)*(k**2)*WG(RG(RL)*k)*W(RL*k)/k, 0.001, kmax)
################################################################################
#Trapezoidal Integrations
rref = 0.073
def ig_sig0(RL,k):
    return Del2k(k)*W(RL*k)**2/k
def ig_sigG(RL,j,k):
    return Del2k(k)*(k**(2*j))*WG(RG(RL)*k)**2/k
def ig_sig1m(RL,R0,k):
    return Del2k(k)*(k**2)*WG(RG(RL)*k)*W(RL*k)/k
def ig_sig1mX(RL,R0,k):
    return Del2k(k)*(k**2)*WG(RG(RL)*k)*W(R0*k)/k
def ig_SX(RL,R0,k):
    return Del2k(k)*W(RL*k)*W(R0*k)/k
def sig0_trapz(RL,kf=50.,N=2000):
    kmax = kf/RL
    K = n.exp(n.linspace(n.log(0.001),n.log(kmax),N))
    Y = ig_sig0(RL,K)
    return n.trapz(Y,K) 
def sigG_trapz(RL,j,kf=100.,N=2000,kmin=0.01):
    kmax = kf/RL
    kmin = kmin/RL
    K = n.linspace(kmin,kmax,N)
    Y = ig_sigG(RL,j,K)
    return n.trapz(Y,K) 
def sig1m_trapz(RL,R0,kf=15.,N=5000,kmin=0.01):
    kmax = kf/RL
    kmin = kmin/R0
    K = n.linspace(kmin,kmax,N)
    Y = ig_sig1m(RL,R0,K)
    return n.trapz(Y,K)
def sig1mX_trapz(RL,R0,kf=15.,N=2000,kmin=0.01):    #further check
    kmax = kf/RL
    kmin = kmin/R0
    K = n.linspace(kmin,kmax,N)
    Y = ig_sig1mX(RL,R0,K)
    return n.trapz(Y,K)
def SX_trapz(RL,R0,kf=6.,N=2000,kmin=0.01): 
    kmax = kf/RL
    kmin = kmin/R0
    K = n.linspace(kmin,kmax,N)
    Y = ig_SX(RL,R0,K)
    return n.trapz(Y,K)
def SXlog_trapz(RL,R0,kf=10.,N=5000,kmin=0.01): 
    kmax = kf/RL
    kmin = kmin/R0
    K = n.exp(n.linspace(n.log(kmin),n.log(kmax),N))
    Y = ig_SX(RL,R0,K)
    return n.trapz(Y,K)
##################################################################################
def plotk_sig0(RL,kf=50.):
    K = n.linspace(0.001,kf,100)
    plt.figure()
    plt.plot(K,ig_sig0(RL,K))
    plt.grid()
    plt.show()
    return
def plotk_sigG(RL,j,kf=100.):
    K = n.linspace(0.001,kf,100)
    plt.figure()
    plt.plot(K,ig_sigG(RL,j,K))
    plt.grid()
    plt.show()
    return
def plotk_SX(RL,R0,kf=10.):
    K = n.linspace(0.001,kf,100)
    plt.figure()
    plt.plot(K,Del2k(K)*W(RL*K)*W(R0*K)/K)
    plt.grid()
    plt.show()
    return
def plotk_sig1mX(RL,R0,kf=100.):
    K = n.linspace(0.001,kf,100)
    plt.figure()
    plt.plot(K,Del2k(K)*(K**2)*WG(RG(RL)*K)*W(R0*K)/K)
    plt.show()
    return
def plotk_sig1m(RL,R0,kf=100.):
    K = n.linspace(0.001,kf,100)
    plt.figure()
    plt.plot(K,Del2k(K)*(K**2)*WG(RG(RL)*K)*W(RL*K)/K)
    plt.show()
    return