import itertools
import numpy as n
import matplotlib.pyplot as plt
import cosmolopy.perturbation as pb
import cosmolopy.density as cd
from scipy.integrate import quad
cosmo = {'baryonic_effects':True,'omega_k_0':0,'omega_M_0':0.315, 'omega_b_0':0.0487, 'n':0.96, 'N_nu':0, 'omega_lambda_0':0.685,'omega_n_0':0., 'sigma_8':0.829,'h':0.673}
def main():
    # Generate Data...
    numdata = 100
    x = n.random.random(numdata)
    y = n.random.random(numdata)
    z = x**2 + y**2 + 3*x**3 + y #+ n.random.random(numdata)

    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)

    # Evaluate it on a grid...
    nx, ny = 20, 20
    xx, yy = n.meshgrid(n.linspace(x.min(), x.max(), nx), 
                         n.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)

    print m
    
    # Plot
    plt.imshow(zz, extent=(x.min(), x.max(), y.min(), y.max()))
    plt.scatter(x, y, c=z)
    plt.show()

    
def mainSX():
    # Generate Data...
    numdata = 100
    logR0min,logR0max = n.log(0.2),n.log(40.)
    logrlmin = n.log(0.04)
    logR0 = logR0min+n.random.random(numdata)*(logR0max-logR0min)
    r,R0 = [],[]
    for lR0 in logR0: 
        lr = logrlmin+n.random.random()*(lR0-logrlmin)
        r.append(n.exp(lr))
        R0.append(n.exp(lR0))
    x,y,z = n.array(r), n.array(R0), n.array([])
    for i, X in enumerate(x):
        print "running %d out of %d" % (i,x.size)
        Z = SX(X,y[i])
        z = n.append(z,Z)
        print "r=%f; R0=%f; z=%f" %(X,y[i],Z)
        #z = n.append(z,sig1mX(X,y[i]))
    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)
    print n.array(m)
    import IPython; IPython.embed()
    # Evaluate it on a grid...
    nx, ny = 20, 20
    xx, yy = n.meshgrid(n.linspace(x.min(), x.max(), nx), 
                         n.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)
    plt.imshow(zz, extent=(x.min(), x.max(), y.min(), y.max()))
    plt.xlabel('RL'); plt.ylabel('R0')
    plt.scatter(x, y, c=z)
    plt.show()

def mainSXlog():
    # Generate Data...
    numdata = 100
    logR0min,logR0max = n.log(0.2),n.log(30.)
    logrlmin = n.log(0.06)
    logR0 = logR0min+n.random.random(numdata)*(logR0max-logR0min)
    logr = []
    for lR0 in logR0: 
        lr = logrlmin+n.random.random()*(lR0-logrlmin)
        logr.append(lr)
    x,y,z = n.array(logr), n.array(logR0), n.array([])
    for i, X in enumerate(x):
        print "running %d out of %d" % (i,x.size)
        ###########################
        Z = sig1mXlog(n.exp(X),n.exp(y[i]))
        ###########################
        z = n.append(z,Z)
        print "r=%f; R0=%f; z=%f" %(n.exp(X),n.exp(y[i]),Z)
        #z = n.append(z,sig1mX(X,y[i]))
    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)
    print n.array(m)
    import IPython; IPython.embed()
    # Evaluate it on a grid...
    nx, ny = 20, 20
    xx, yy = n.meshgrid(n.linspace(x.min(), x.max(), nx), 
                         n.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)
    plt.imshow(zz, extent=(x.min(), x.max(), y.min(), y.max()))
    plt.xlabel('RL'); plt.ylabel('R0')
    plt.scatter(x, y, c=z)
    plt.show()

def SX(RL,R0,kf=10.): 
    kmax = kf/R0
    return quad(lambda k: Del2k(k)*W(RL*k)*W(R0*k)/k, 0, kmax)[0]
def sig1mX(RL,R0,kf=10.):
    kmax = kf/R0
    return quad(lambda k: Del2k(k)*(k**2)*WG(RG(RL)*k)*W(R0*k)/k, 0, kmax)[0]
def SXlog(RL,R0,kf=10.): 
    logmax = n.log(kf/R0)
    return quad(lambda logk: Del2k(n.exp(logk))*W(RL*n.exp(logk))*W(R0*n.exp(logk)), 0, logmax)[0]
def sig1mXlog(RL,R0,kf=10.):
    logmax = n.log(kf/R0)
    return quad(lambda logk: Del2k(n.exp(logk))*(n.exp(logk)**2)*WG(RG(RL)*n.exp(logk))*W(R0*n.exp(logk)), 0, logmax)[0]
def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = n.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = n.linalg.lstsq(G, z)
    return m

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
#mainSXrand()
mainSXlog()



