#!/usr/bin/env python

import numpy as np
from optparse import OptionParser

import os

from Utils import data_process_utils as dpu
from Utils import exp_opt_utilsMEM as eou

import matplotlib.pyplot as plt
#from matplotlib import gridspec


parser = OptionParser()
parser.add_option("-s", 
                  action="store",
                  type = "float",
                  dest = "s")
parser.add_option("-l", "--load",
                  action="store",
                  type = "int",
                  dest = "load")
parser.add_option("-c", "--cost",
                  action="store",
                  type = "int",
                  dest = "cost")
options = parser.parse_args()[0]
s_rat = options.s
load_mat = options.load
cost = options.cost

# load transmission matrices
print "Loading transmission matrices"
#file_mat = '0923_126lambda_208k'
#file_mat = '0925_101lambda_256k'
#file_dir = '../Data-Numerics/' + file_mat + '/'

#file_mat = 'test'
file_mat = '151k_16m_30s'
#file_mat = '301k_16m_30s'
#file_mat = '601k_16m_30s'
#file_mat = '1201k_16m_30s'

file_dir = '../Data-Random/' + file_mat + '/'

t = np.load(file_dir + file_mat+'_filter.npy')
lambdas = np.load(file_dir + file_mat+'_wavelength.npy')

#t = t[:,40:61,:]

print "Received transmission matrices of dimensions:"
print "# pixels: %i \n# input angles: %i \n# wavelengths: %i" % np.shape(t) 

N_pix, N_k, N_l = np.shape(t)
n_pix = np.sqrt(N_pix)
pic_shape = (n_pix, n_pix)

l_range = lambdas[-1] - lambdas[0]
dl = lambdas[1] - lambdas[0]

indx_0 = int(np.floor(N_l / 2))
lambda_0 = lambdas[indx_0]


# parameters
nloop = 5
#s_rat = 0.01
s = s_rat * l_range
t0 = 0.0
ds0 = 1e-3
#load_mat = True
file_out = file_dir + "cost=%i/" % cost + file_mat  + "_%.3f" % s_rat
if (not os.path.isdir(file_dir + "cost=%i/" % cost)):
    os.makedirs(file_dir + "cost=%i/" % cost)
    

# weighting function for spatial average
u, v = np.load(file_dir + file_mat + '_singmats.npy')
weights = np.sum(np.abs(u)**2, axis=1)
weights /= np.sum(weights)
###
weights = np.ones(np.shape(weights))
weights /= np.sum(weights)
###


print "Calculating envelope"
phi = eou.calc_phi_gauss(lambdas, lambda_0, t0, s)
#phi = eou.calc_phi_fermi(lambdas - lambda_0, (s, l_range * 0.020))
#phi = eou.calc_phi_fermi(lambdas - lambda_0, (s, l_range * 0.005))

plt.plot(lambdas, np.abs(phi)**2, 'o-r')
plt.savefig(file_out + '_opt_envelope.png')
plt.clf()

print "Initializing cost function"
F = eou.Functional(phi, t, lambdas, lambda_0, weights, file_out, load_mat, cost)
if (cost == 1):
    L = F.L_long
    G = F.G_long
if (cost == 2):
    L = F.L_trans
    G = F.G_trans
if (cost == 3):
    L = F.L_long_FT
    G = F.G_long_FT
if (cost == 4):
    L = F.L_long_Abs
    G = F.G_long_Abs

# initial guess
phat_rand = (np.random.random(size=(N_k)) - 0.5) - 1.0j * (np.random.random(size=(N_k)) - 0.5)

# minimize functional 
phat = eou.optimize(L, G, phat_rand, ds0, nloop)

print "Saving optimal input"
np.save(file_out+'_opt_input', phat)
