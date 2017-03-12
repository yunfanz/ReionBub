import itertools
import numpy as n
import matplotlib.pyplot as plt
import cosmolopy.perturbation as pb
import cosmolopy.density as cd
from scipy.integrate import quad
from scipy.interpolate import interp1d, interp2d
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

def SXlog(RL,R0,kf=10.): 
    logmax = n.log(kf/R0)
    return quad(lambda logk: Del2k(n.exp(logk))*W(RL*n.exp(logk))*W(R0*n.exp(logk)), -1000., logmax)[0]

def SX(RL,R0,kf=10.): 
    kmax = kf/R0
    return quad(lambda k: Del2k(k)*W(RL*k)*W(R0*k)/k, 0.001, kmax)[0]
def sig1mX(RL,R0,kf=10.):
    kmax = kf/R0
    return quad(lambda k: Del2k(k)*(k**2)*WG(RG(RL)*k)*W(R0*k)/k, 0.001, kmax)[0]

# lR0min, lR0max = n.log(0.2),n.log(40.)
# lrlmin = n.log(0.04)
# lR0 = n.linspace(lR0min, lR0max,50)
# arr = []
# lRl = []
# for lr0 in lR0:
#     lrl = n.arange(lrlmin,lr0,0.5)
#     lrl = n.append(lrl,lr0)
#     #lrl = n.arange(lrlmin,lR0max,0.5)
#     ar = []
#     for ll in lrl: 
#         rl,r0 = n.exp(ll), n.exp(lr0)
#         print rl, r0
#         ar.append(SX(rl,r0)) 
#     arr.append(ar)
#     lRl.append(lrl)
# n.savez('SX',n.array(lRl),n.array(lR0),n.arrar(arr))

lR0min, lR0max = n.log(0.2),n.log(40.)
lrlmin = n.log(0.04)
lR0 = n.linspace(lR0min, lR0max,50)
lRl = n.linspace(lrlmin,lR0max,50)
xx,yy = n.meshgrid(lRl,lR0)
arr = n.ones_like(xx)
for i,lrl in enumerate(lRl):
    for j,lr0 in enumerate(lR0):
        rl,r0 = n.exp(lrl),n.exp(lr0)
        arr[i,j] = sig1mX(rl,r0)
        print rl,r0,arr[i,j]

# plt.figure()
# plt.imshow(arr)
# plt.colorbar()
# plt.show()
n.savez('logsig1mX',xx,yy,arr)
fsig1mX = interp2d(xx,yy,arr,kind='cubic')
print sig1mX(0.1,0.2), fsig1mX(n.log(0.1),n.log(0.2))
print sig1mX(0.7,10.), fsig1mX(n.log(0.7),n.log(10.))