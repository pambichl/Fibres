#!/usr/bin/env python

import numpy as np


print "Loading transmission matrices"
#file_mat = '0917_401lambda_1k'
#file_mat = '101_wavelength'
#file_mat = 'TransmissionMatrix'
#file_mat = '0922_121lambda_121k'
#file_mat = 'TM_20_15_10'
file_mat = '0923_126lambda_208k'


t = np.load(file_mat+'_filter.npy')
lambdas = np.load(file_mat+'_wavelength.npy')

N_l = np.shape(lambdas)[0]
indx_0 = int(np.floor(N_l / 2))
t_0 = t[:,:,indx_0]


print "Calculating singular value decomposition"
U, S, V = np.linalg.svd(t_0)


print "Saving data to file"
np.save(file_mat+'_svd', (U, S, V))

