#!/usr/bin/env python

import numpy as np
import scipy.optimize as spo

import matplotlib.pyplot as plt

import subprocess
import os
from optparse import OptionParser

from Utils import utils as ut
from Scattering import scattering2D as scat

from Packets import transmission as trans
from Optimize import functionals as func

Pi = np.pi
I = 0. + 1.J


# read in options from command line
parser = OptionParser()
parser.add_option("-n", "--nloop",
                  action="store",
                  type = "int",
                  dest = "nloop")
parser.add_option("-i", "--inputtime",
                  action="store",
                  type = "float",
                  dest = "t0")
parser.add_option("-s", "--inputstd",
                  action="store",
                  type = "float",
                  dest = "ts")
parser.add_option("-q", "--qeigst",
                  action="store",
                  type = "int",
                  dest = "qeigst")
parser.add_option("-r", "--readst",
                  action="store",
                  type = "int",
                  dest = "readst")
parser.add_option("-e", "--randseed",
                  action="store",
                  type = "int",
                  dest = "randseed")
parser.add_option("--trans",
                  action="store",
                  type = "int",
                  dest = "opt_trans")
parser.add_option("-t", "--tag",
                  action="store",
                  type = "str",
                  dest = "tag")
parser.add_option("-o", "--noise",
                  action="store",
                  type = "float",
                  dest = "noiserat")
parser.add_option("-p", "--randphase",
                  action="store",
                  type = "float",
                  dest = "randphase")
parser.add_option("--auto",
                  action="store",
                  type = "int",
                  dest = "auto")
options = parser.parse_args()[0]
nloop = options.nloop
t0 = options.t0
s = options.ts
qeigst = options.qeigst
readst = options.readst
randseed = options.randseed
opt_trans = options.opt_trans
tag = options.tag
noiserat = options.noiserat
randphaseamp = options.randphase
auto = options.auto
if (not qeigst): qeigst = 0 # if negative start with random vector
if (randseed): np.random.seed(randseed)
if (not t0): t0 = 20.0/(2.0*s)
if (s): s = 0.5/s
if (not s): s = (wlist[-2] - wlist[1]) / 6.0
if (not tag): tag = ""
if (not noiserat): noiserat=0.0
if (not randphaseamp): randphaseamp=0.0


# read in parameters from xml file
par = ut.read_xml()
try:
   filen = par['file'] # namestring of calculation where scatter data is stored
   modes_min = float(par['modes_min'])
   modes_max = float(par['modes_max'])
   refmodes = int(par['refmodes']) # number of modes to be considered
   W = float(par['W'])
   pphwl = float(par['points_per_hwl'])
except KeyError:
    raw_input("WARNING: parameter missing in Abmessung.xml")


# read in scattering matrices
data_direc = ("/home/ambichl/Universitaet/Scattering-Data/"
              "VSC2/Fibres-20130802/" + filen + "/")

print "Reading scattering matrices"
if (not os.path.exists("scatter."+filen+".dat.npy")):
   S, dims, energs = scat.read_S(data_direc, filen, old_ver=0)
   np.save("scatter."+filen+".dat", (S, dims, energs))
else:
   S, dims, energs = np.load("scatter."+filen+".dat.npy")


# other parameters
pixel = True # whether system is transformed to y-eigenbasis
#pixel = False # whether system is transformed to y-eigenbasis
Nav = 1 # number of random noise configurations to be averaged
#phase_only = False # whether optimization w.r.t. phase only
phase_only = True # whether optimization w.r.t. phase only
#ds0 = 5.0 # step size for gradient in optimization
ds0 = 1e-2
#ds0 = 0.001
proj_list = np.zeros((refmodes,), dtype="complex")
proj_list[:10] = 1.0


kmean = 0.5 * (modes_max + modes_min) * np.pi # lead_width assumed to be 1
dk =  kmean * 10**(-8)
nin_Max = int(0.5 * (modes_max + modes_min)) # nin_Max = n_open_modes


wlist = np.sqrt(2.0 * np.array(energs, dtype="float"))
w0 = 0.5 * (wlist[-2] + wlist[1])
dw = w0 * 10**(-8)
refpos = int(0.5 * len(wlist) / 3)


nyout = int(0.5*(modes_min+modes_max)) * pphwl
yvals = np.linspace(0.0,1.0,nyout+2)
ky = (np.arange(refmodes)+1)*np.pi
kx = np.array([np.sqrt(w**2 - ky**2) for w in wlist[1::3]])
lead_funcs = np.sqrt(2.) * np.sin(np.einsum('m,y->ym', ky, yvals))
chi = np.einsum('wm,ym->wym', 1.0/np.sqrt(kx), lead_funcs)

y_op = np.einsum('ym,y,yn->mn', lead_funcs.conj(), yvals, lead_funcs)
y_eigvec = ut.sort_eig(y_op)[1]
y_funcs = np.dot(lead_funcs, ut.sort_eig(y_op)[1]).T
if (pixel): 
   chi = np.einsum('ym,wm,mn->wyn', lead_funcs, 1.0/np.sqrt(kx), y_eigvec)


dummy, t_q = trans.calc_t(S, dims, refmodes)
if (pixel):
   t_q = np.einsum('ij,wik,kl->wjl', y_eigvec.conj(), t_q, y_eigvec)
dt = (t_q[2::3] - t_q[0::3]) / (2*dk)
t = t_q[1::3]


# add random noise with given strength
print "Adding noise"
noisamp = noiserat * np.abs(t_q)
noise_full = np.zeros(np.shape(t_q), dtype="complex")
for i in range(Nav):
   noise = noisamp * np.random.random(np.shape(t_q)) * np.exp(I * 2.0 * np.pi * np.random.random(np.shape(t_q)))
   noise_full += noise / Nav

t_noise_full = t_q + noise_full
t_noise = t_noise_full[1::3]


def calc_phi_gauss(wlist, w0, t0, s):
   '''
   Calculates Gaussian amplitude function for 
   a packet with center frequency w0,
   standard deviation in frequency space 2*s,
   and peak at time t=t0.
   '''
   return (1.0 / (2.0 * Pi))**(1.0/4.0) * \
       1.0 / np.sqrt(s) * \
       np.exp( -((wlist - w0) / (2.0 * s))**2 ) * \
       np.exp( I * t0 * (wlist - w0) )


def calc_phi_uniform(wlist, w0, t0, s):
   '''
   Calculates uniform amplitude function for 
   a packet with center frequency w0,
   width in frequency space 2*s,
   and peak at time t=t0.
   '''
   uniform_mask = np.logical_and(wlist >= w0-s, wlist <= w0+s)

   phi = 1.0/(2.0*s) * np.exp( I * t0 * (wlist - w0) )
   phi[~uniform_mask] = 0.0

   return phi


phi_full = calc_phi_gauss(wlist, w0, t0, s)
#phi_full = calc_phi_uniform(wlist, w0, t0, s)
phi = phi_full[1::3]  
Nphi = np.dot(phi.conj(), phi) * (wlist[4] - wlist[1])
dphi = (phi_full[2::3] - phi_full[0::3]) / (2*dw)

phit_full = np.einsum('i,ijk->ijk', phi_full, t_q)
dphit = (phit_full[2::3] - phit_full[0::3]) / (2*dw)
phit = phit_full[1::3]

phit_noise_full = np.einsum('i,ijk->ijk', phi_full, t_noise_full)
phit_noise = phit_noise_full[1::3]

Dw = (wlist[4] - wlist[1]) / Nphi

 
F = func.Functional(phi, t_noise, phit_noise_full, wlist, Dw, opt_trans, proj_list)
if (opt_trans == 1):
   L_choice = F.L_corr_w
   G_choice = F.G_corr_w
elif (opt_trans == 2):
   L_choice = F.L_corr_t
   G_choice = F.G_corr_t
elif (opt_trans == 3):
   L_choice = F.L_trans
   G_choice = F.G_trans
elif (opt_trans == 4):
   L_choice = F.L_proj
   G_choice = F.G_proj
else:
   L_choice = F.L_long
   G_choice = F.G_long


# Gaussian intensity profile for input pixels
ylocs = (W * np.arange(0,refmodes) + 0.5) / refmodes
pix_intensities = np.exp(-0.5*(ylocs-0.5*W)**2/(0.3*W)**2)

def normalize(vec):
   '''Normalizes vector in the proper way
   for optimization.'''

   global phase_only, pixel, pix_intensities

   if (phase_only and pixel):
      vec = vec / np.absolute(vec) * pix_intensities
   elif (phase_only and not pixel):
      vec = vec / np.absolute(vec)  

   vec = vec / np.linalg.norm(vec) 
 
   return vec


# take only transmitted states for q
t_ref = t_noise[refpos]
tdt_eigval, tdt_eigvec = ut.sort_eig(t_ref.T.conj().dot(t_ref))
trans_mask = np.abs(tdt_eigval) > 1.0e-3
trans_vec = tdt_eigvec[:,trans_mask]

print
print tdt_eigval
print

q_arg = np.einsum('ij,wjk,kl->wil',
                  trans_vec.conj().T, t_noise_full[3*refpos:3*refpos+3], trans_vec)

q = scat.calc_Q(q_arg, dw, inv=True)[0]
q = trans_vec.dot(q).dot(trans_vec.conj().T)
qeigstates = ut.sort_eig(q)[1].T


phat = qeigstates[qeigst]

phat_rand = (np.random.random(size=(refmodes)) - 0.5) - I * (np.random.random(size=(refmodes)) - 0.5)
phat_rand = phat_rand / np.linalg.norm(phat_rand)

if(qeigst<0): phat = phat_rand
if(readst): phat = np.load("movie."+filen+"."+"opt_rand"+".dat.npy")[2]

if (not nloop and nloop!=0): nloop = 100000
phat = normalize(phat)
phat_new = phat
phat_old = phat
phat_min = phat
i_min = 0


# v0 = phat
# print np.dot(v0.conj(), G_choice(phat))


### minimize chosen spread ###

#L_old = np.absolute(L_choice(phat_old))
L_old = L_choice(phat_old) 
L_min = L_old
print "step: %5i of %5i gradient: %4.6f %4.6f" % (0, nloop, L_old.real, np.linalg.norm(G_choice(phat_old)))
for i in range(nloop):

   # calculate what value of functional would be if step would be made
   phat_old = normalize(phat_old) 
   L_old = L_choice(phat_old)   
   ds = ds0 
   phat_int = phat_old - G_choice(phat_old) * ds
   phat_int = normalize(phat_int) 
   L_int = L_choice(phat_int) 
   
   # if new value is larger than old one reduce step size until it is smaller
   count = 0
   while (L_old < L_int and abs((L_old - L_int) / L_old) > 0.00001 and count <= 5000):
   #while (L_old < L_int and count <= 5000):
      ds = 0.50*ds
      phat_int = phat_old - G_choice(phat_old) * ds
      phat_int = normalize(phat_int) 
      L_int = L_choice(phat_int)
      AbsGrad = np.linalg.norm(G_choice(phat_int))
      count += 1

   # finally perform step with reduced step size
   phat_new = phat_old - G_choice(phat_old) * ds
   phat_new = normalize(phat_new) 
   L_new = L_choice(phat_new)

   if (L_new < L_min):
      L_min = L_new
      phat_min = phat_new
      i_min = i

   AbsGrad = np.linalg.norm(G_choice(phat_new))
   print "step: %5i of %5i gradient: %4.10f %4.10f" % (i+1, nloop, L_new.real, AbsGrad)

   phat_old = phat_new    

print "found minimum of functional and gradient at step %i: %4.6f, %4.6f" % (i_min+1, L_min.real, np.linalg.norm(G_choice(phat_min)))
phat = phat_min
# if (nloop==0):
#    phat /= np.linalg.norm(phat)
# else:
#    phat = normalize(phat)
phat = normalize(phat)

# phat_min = spo.minimize(L_choice, phat_rand, method='Newton-CG', jac=G_choice).x
# phat = phat_min / np.linalg.norm(phat_min)
# print L_choice(phat).real, np.linalg.norm(G_choice(phat))

a, c, d = F.calc_coeffs(phat)
varc_out = a/d - (c/d)**2

# add random phases to incoming wave packet at each frequency
randphase = np.exp(I * (-1)**np.random.random_integers(0, 1, np.shape(phi))
                   * randphaseamp * np.pi * np.random.random(np.shape(phi)))
randphase_full = np.reshape(np.transpose((randphase,randphase,randphase), axes=(1,0)), np.shape(phi_full))
phi_rand_full = np.einsum('w,w->w', randphase_full, phi_full)
phit_rand_full = np.einsum('w,wmn->wmn', randphase_full, phit_full)


# save data needed for time-dependent calculations and movies
np.save("movie."+filen+"."+tag+".dat", (phi_rand_full, phit_rand_full, phat,
                                        t0, 1.0/(2.0*s), refpos, varc_out, wlist, chi, yvals))


print
print "Absolute values of input coefficients:"
print np.absolute(phat)
print


print
if (noiserat!=0.0):
   print "Average signal-to-noise-ratio = %f" % np.mean(np.abs(t_q / noise_full))
print "Average noise-to-signal-ratio = %f" % np.mean(np.abs(noise_full / t_q))
print


### ATTENTION: may lead to numerical errors if first component of vectors is zero ### 
print
print "Differences of q-eigenstates to final state:"
for vec in qeigstates:
   print np.linalg.norm(vec*np.exp(-I*np.angle(vec[0])) - phat*np.exp(-I*np.angle(phat[0])))
print


if (auto):
   if (not os.path.exists(filen+".opt.log")):
      with open(filen+".opt.log", "w") as fstream:
         fstream.write(" %6f" % AbsGrad)
   else:
      with open(filen+".opt.log", "a") as fstream:
         fstream.write(" %6f" % AbsGrad)


rand_arr = np.random.random(10) * np.exp(I * 2.0 * np.pi * np.random.random(10))

print np.abs(np.mean(rand_arr))
print np.mean(np.abs(rand_arr))
