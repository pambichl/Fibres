#!/usr/bin/env python

import numpy as np

from Utils import exp_opt_utilsMEM as eou

file_mat = '0925_101lambda_256k'
file_dir = '../Data-Numerics/' + file_mat + '/'

#file_mat = '151k_16m_30s'
#file_mat = '301k_16m_30s'
#file_mat = '601k_16m_30s'
#file_mat = '1201k_16m_30s'

#file_dir = '../Data-Random/' + file_mat + '/'


print "Loading transmission matrices"
t = np.load(file_dir + file_mat + '_filter.npy')
N_x, N_k, N_l = np.shape(t)

N_r = 10000

print "Calculating correlations"
rand_corr = np.zeros((N_r,N_l))
for r in range(N_r):
    print "Calculating output #%i of %i" % (r+1,N_r) 
    rand_in = (np.random.random(size=(N_k)) - 0.5) - 1.0j * (np.random.random(size=(N_k)) - 0.5)
    rand_out = np.einsum('xkl,k->xl', t, rand_in)
    rand_corr[r] = eou.calc_corr_vec(rand_out)

np.save(file_dir + file_mat + '_randsCorr', rand_corr)

