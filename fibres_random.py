#!/usr/bin/env python

import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt
import sys
import os
from optparse import OptionParser

# read in options from command line
parser = OptionParser()
parser.add_option("-n", "--rindx",
                  action="store",
                  type = "float",
                  dest = "n")
parser.add_option("-r", "--radius",
                  action="store",
                  type = "float",
                  dest = "r")
parser.add_option("-l", "--length",
                  action="store",
                  type = "float",
                  dest = "L")
parser.add_option("-m", "--Nm",
                  action="store",
                  type = "int",
                  dest = "Nm")
parser.add_option("--kmin",
                  action="store",
                  type = "float",
                  dest = "kmin")
parser.add_option("--kmax",
                  action="store",
                  type = "float",
                  dest = "kmax")
parser.add_option("-k", "--Nk",
                  action="store",
                  type = "int",
                  dest = "Nk")
parser.add_option("-s", "--Ns",
                  action="store",
                  type = "int",
                  dest = "Ns")
parser.add_option("--seed",
                  action="store",
                  type = "int",
                  dest = "randseed")
options = parser.parse_args()[0]
n = options.n
r = options.r
Ltot = options.L
Nm = options.Nm
kmin = options.kmin
kmax = options.kmax 
Nk = options.Nk
Ns =  options.Ns
randseed = options.randseed
np.random.seed(randseed)

L = Ltot / Ns

spec_string = "%ik_%im_%is" % (Nk,Nm,Ns)
#spec_string = "test"
dir_out = "../Data-Random/" + spec_string

if (np.sqrt(Nm) != np.floor(np.sqrt(Nm))):
    print "WARNING: number of modes not a square number"

# logfile
f = open(spec_string + ".txt", "w")
f.write("refractive index of fiber n = %.3f \n" % n)
f.write("radius of fiber           r = %.3f \n" % r)
f.write("length of fiber           L = %.3f \n" % L)
f.write("number of segments        N = %i \n" % Ns)
f.write("minimum wavenumber     kmin = %.3f \n" % kmin)
f.write("maximum wavenumber     kmax = %.3f \n" % kmax)
f.write("random seed           srand = %i   \n" % randseed)



def det_A(b, k, n, r):
    '''
    Determinant of the coefficient matrix corresponding to
    a fiber with 1D cross section.
    parameters:

    b: float
    propagation constant in z-direction
    k: float
    total wavenumber
    n: float
    refractive index of the fiber core
    r: float
    radius of the fiber core
    '''

    from cmath import sqrt

    kappa = sqrt(b**2 - k**2)
    ky = sqrt(k**2 * n**2 - b**2)

    s1 =  np.exp(-2.0 * 1.0j * r * ky - 2.0 * r * kappa) * k**2
    s2 = -np.exp( 2.0 * 1.0j * r * ky - 2.0 * r * kappa) * k**2
    s3 = -np.exp(-2.0 * 1.0j * r * ky - 2.0 * r * kappa) * k**2 * n**2
    s4 =  np.exp( 2.0 * 1.0j * r * ky - 2.0 * r * kappa) * k**2 * n**2 

    return s1 + s2 + s3 + s4

def calc_beta(k, n, r):
    '''
    Calculate propagation constants in z-direction.
    We are only interested in solutions bounded in
    transverse direction and propagating in z-direction,
    so k < beta < n * k.
    '''

    x_steps = 100000
    x_vals = np.linspace(k, n*k, x_steps, endpoint=False)

    det_norms = np.abs(np.vectorize(det_A)(x_vals, k, n, r))

    beta = [0.0]
    for i in range(1,len(x_vals)-1):
        val_l = det_norms[i-1]
        val_r = det_norms[i+1]
        
        val_c = det_norms[i]

        if (val_c < val_l and val_c < val_r):
            beta = np.append(beta, x_vals[i])
            
    beta = beta[1:]
    return np.array(beta)[::-1]


k_vals = np.linspace(kmin, kmax, Nk)

prop_perf = np.zeros((Nm,Nm,Nk), dtype="complex")
# diagonal matrix with propagation phase factors
Nb = np.zeros((Nk), dtype="int")
print "Calculating propagation constants"
for i_k,k in enumerate(k_vals):
    beta = calc_beta(k, n, r)
    Nb[i_k] = np.shape(beta)[0]
    if (Nb[i_k] < Nm):
        print "Less (%i) than desired number (%i) of modes open!" % (Nb[i_k], Nm)
        sys.exit()
    print "k# %i of %i, " % (i_k+1,Nk) + "# of open modes: %i(%i)" % (Nb[i_k], Nm)
    prop_perf[:,:,i_k] = np.diag(np.exp(-1.0j * beta[:Nm] * L))
    
prop_rand = np.random.normal(size=(Nm,Nm,Ns)) + 1.0j * np.random.normal(size=(Nm,Nm,Ns))
# full random matrix with complex coupling coefficients
# of same size as propagation matrices
# Ns different matrices for Ns random segments
for i in range(Ns):
    prop_rand[:,:,i] = 0.5 * (prop_rand[:,:,i].conj().T + prop_rand[:,:,i])
    prop_rand[:,:,i] = np.linalg.eig(prop_rand[:,:,i])[1]

t = np.einsum('pq,qmk->pmk', prop_rand[:,:,0], prop_perf)
for s in range(1,Ns): 
    t = np.einsum('pq,qmk,mnk->pnk', prop_rand[:,:,i], prop_perf, t)

if (not os.path.exists(dir_out)):
    os.makedirs(dir_out)

np.save(dir_out + "/" + spec_string + "_wavelength", k_vals)
np.save(dir_out + "/" + spec_string + "_filter", t)

i0 = np.floor(0.5 * Nk)
dk = k_vals[1] - k_vals[0]
q = 1.0j * np.linalg.inv(t[:,:,i0]).dot(t[:,:,i0+1] - t[:,:,i0-1]) / (2.0 * dk)
np.save(spec_string + "_q", q)

u,s,v = np.linalg.svd(t[:,:,i0])
v = v.conj().T
np.save(dir_out + "/" + spec_string + "_singmats", (u,v))




