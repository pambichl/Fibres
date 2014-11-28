#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.io
import copy

import data_process_utils as dpu

import matplotlib.pyplot as plt
#from matplotlib import gridspec


# load transmission matrices
print "Loading transmission matrices"
#file_mat = '0917_401lambda_1k'
#file_mat = '101_wavelength'
#file_mat = 'TransmissionMatrix'
#file_mat = '0922_121lambda_121k'
#file_mat = 'TM_20_15_10'
#file_mat = '0923_126lambda_208k'
file_mat = '0925_101lambda_256k'

file_dir = '/home/ambichl/Universitaet/Fibres-Hui/Experiments/Data-Numerics/' + file_mat + '/'

print "Trying to load with NumPy:",
t = np.load(file_dir + file_mat + '.npy')
lambdas = np.load(file_dir + file_mat + '_wavelength.npy')
print "succeded."


# # Calculate which pixels to keep and which are dark ones
# print "Cutting away dark pixels"
# t_amp = np.abs(t)
# int_mean = np.mean(t_amp**2, axis=(1,2))
# av_int_mean = np.mean(int_mean)
# dark_mask = int_mean < 0.15 * av_int_mean


#t = t[:,49:52,:]

print "Received transmission matrices of dimensions:"
print "# pixels: %i \n# input angles: %i \n# wavelengths: %i" % np.shape(t)  


# Calculate all needed ranges etc.
N_pix, N_k, N_l = np.shape(t)
n_pix = np.sqrt(N_pix)
pic_shape = (n_pix, n_pix)

l_range = lambdas[-1] - lambdas[0]
dl = lambdas[1] - lambdas[0]

lambdas_shift = np.arange(N_l) * dl

w = np.linspace(0.0, 2.0*np.pi/dl, N_l)
w_range = w[-1] - w[0]
dw = 2.0 * np.pi / l_range


# plots for checking
pixel_choice = 50 + n_pix * 50 
k_choice = int(0.5 * (N_k - np.sqrt(N_k)))
l_choice = int(0.5 * N_l)
pic_shape = (n_pix, n_pix)
sup = (np.random.random((N_k)) - 0.5)+ 1.0j * (np.random.random((N_k)) - 0.5)
sup /= np.linalg.norm(sup)
np.save(file_dir + file_mat+'_randinp', sup)

# speckle
out_vec = np.einsum('xkl,k->xl', t, sup)[:,l_choice]
out_pic = np.reshape(out_vec, pic_shape)
plt.xlim(0,n_pix)
plt.ylim(0,n_pix)
plt.pcolor(np.abs(out_pic))
plt.savefig(file_dir + file_mat+'.png')


# perform Fourier transform
print "Performing Fourier transform"
spec = np.zeros((N_l,), dtype='double')
for k in range(N_k):
    print "input angle %i of %i" % (k+1,N_k)
    t_k_ft = np.fft.fftshift(np.fft.fft(t[:,k,:], axis=1), axes=1)
    spec += np.mean(np.abs(t_k_ft)**2, axis=0)

Norm_spec = np.sum(spec)
spec /= Norm_spec

indx_roll = 0

for i in range(15):
    w_mean = np.einsum('w,w', w, spec)
    indx_roll_inc = int((np.mean(w) - w_mean) / dw)
    indx_roll += indx_roll_inc
    spec = np.roll(spec, indx_roll_inc, axis=0)

print
print "center of mass of average power spectrum: ", w_mean
print "center of wavelength frequencies: ", np.mean(w)
print "step size in Fourier space: ", dw
print 

print "Rolling data and performing inverse Fourier transform"
for k in range(N_k):
    print "input angle %i of %i" % (k+1,N_k)
    t_k_ft = np.fft.fftshift(np.fft.fft(t[:,k,:], axis=1), axes=1)
    t_k_ft_roll = np.roll(t_k_ft, indx_roll, axis=1)
    t_k_roll = np.fft.ifft(np.fft.ifftshift(t_k_ft_roll, axes=1), axis=1)
    t[:,k,:] = t_k_roll

print "Saving data"
np.save(file_dir + file_mat + '_roll', t)
