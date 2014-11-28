#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.io
import copy

import data_process_utils as dpu

import matplotlib.pyplot as plt


# load transmission matrices
print "Loading transmission matrices"
file_mat = '0925_101lambda_256k'

file_dir = '/home/ambichl/Universitaet/Fibres-Hui/Experiments/Data-Numerics/' + file_mat + '/'

print "Trying to load with NumPy:",
t = np.load(file_dir + file_mat + '_roll.npy')
lambdas = np.load(file_dir + file_mat + '_wavelength.npy')
print "succeded."

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
sup = np.load(file_dir + file_mat + '_randinp.npy')

# amp
plt.figure(0)
plt.plot(lambdas, np.abs(t[pixel_choice,k_choice,:]), '-b', lw=1.0)
# phase
plt.figure(1)
plt.plot(lambdas, np.unwrap(np.angle(t[pixel_choice,k_choice,:])), '-b', lw=1.0)
# speckle
plt.figure(2)
out_vec = np.einsum('xkl,k->xl', t, sup)[:,l_choice]
out_pic = np.reshape(out_vec, pic_shape)
plt.xlim(0,n_pix)
plt.ylim(0,n_pix)
plt.pcolor(np.abs(out_pic))
plt.savefig(file_dir + file_mat+'_recovery.png')
plt.clf()

filter_params_amp = (0.15 * w_range, w_range/300.0)
filter_params_phase = (0.15 * w_range, w_range/300.0)

print
print "Filtering data"
print "Ratio of filter width to spectrum (amplitude): %.2f" % (2.0 * filter_params_amp[0] / (w[-1] - w[0]))
print "Ratio of filter width to spectrum (phase): %.2f" % (2.0 * filter_params_phase[0] / (w[-1] - w[0]))
print

for k in range(N_k):
    print "input angle %i of %i" % (k+1,N_k)

    # separate rolled data in absolute value and phase
    t_amp = np.abs(t[:,k:k+1,:])
    t_phase = np.unwrap(np.angle(t[:,k:k+1,:]))

    # perform Fourier transform of absolute value and phase
    t_amp_ft = np.fft.fftshift(np.fft.fft(t_amp, axis=2), axes=2)
    t_phase_ft = np.fft.fftshift(np.fft.fft(t_phase, axis=2), axes=2)
    
    t_amp_ft_filter = dpu.low_pass_filter(t_amp_ft,
                                          #w - av_amp_ft,
                                          w - np.mean(w),
                                          dpu.fermi,
                                          filter_params_amp,
                                          False)

    t_phase_ft_filter = dpu.low_pass_filter(t_phase_ft,
                                            #w - av_phase_ft,
                                            w - np.mean(w),
                                            dpu.fermi,
                                            filter_params_phase,
                                            False)


    # perform inverse Fourier transform of filtered data
    t_amp_filter = dpu.norm_ifft_shift(t_amp, t_amp_ft_filter)
    t_phase_filter = dpu.norm_ifft_shift(t_phase, t_phase_ft_filter)
    
    t_filter = t_amp_filter * np.exp(1.0j * t_phase_filter)
    t[:,k,:] = t_filter[:,0,:]

print "Saving data"
np.save(file_dir + file_mat+'_filter', t)


# plots for checking

# amp
plt.figure(0)
plt.plot(lambdas, np.abs(t[pixel_choice,k_choice,:]), 'd-r', lw=1.0)
plt.savefig(file_dir + file_mat+'_amp_check.png')
# phase
plt.figure(1)
plt.plot(lambdas, np.unwrap(np.angle(t[pixel_choice,k_choice,:])), 'd-r', lw=1.0)
plt.savefig(file_dir + file_mat+'_phase_check.png')
# speckle
plt.figure(2)
out_vec = np.einsum('xkl,k->xl', t, sup)[:,l_choice]
out_pic = np.reshape(out_vec, pic_shape)
plt.xlim(0,n_pix)
plt.ylim(0,n_pix)
plt.pcolor(np.abs(out_pic))
plt.savefig(file_dir + file_mat+'_recovery_filter.png')

