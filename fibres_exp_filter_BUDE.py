#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.io
#import h5py
import copy

import data_process_utils as dpu

import matplotlib.pyplot as plt
from matplotlib import gridspec


# load transmission matrices
print "Loading transmission matrices"
#file_mat = '0917_401lambda_1k'
#file_mat = '101_wavelength'
#file_mat = 'TransmissionMatrix'
#file_mat = '0922_121lambda_121k'
#file_mat = 'TM_20_15_10'
file_mat = '0923_126lambda_208k'

try:
    print "Trying to load with NumPy:",
    t = np.load(file_mat+'.npy')
    lambdas = np.load(file_mat+'_wavelength.npy')
    print "succeded."
except:
    try:
        print "failed."
        print "Trying to load with SciPy from Matlab:",
        data = scipy.io.loadmat('../Wen/'+file_mat+'.mat')
        t = data['TM']
        lambdas = data['wavelength'][0,:]
        print "succeded."
        np.save(file_mat, t)
    except NotImplementedError:
        print "failed."
        print "Trying to load with H5PY from Matlab:",
        data = h5py.File('../Wen/'+file_mat+'.mat', 'r')
        t_data = np.asarray(data['TM'])
        t_data = np.swapaxes(t_data,0,2)
        t = np.asarray(t_data['real'] + 1.0j * t_data['imag'], dtype='complex') 
        lambdas = np.asarray(data['wavelength'], dtype="float")[:,0]
        data.close()
        print "succeded."
        np.save(file_mat, t)
        np.save(file_mat+'_wavelength', lambdas)
        # it does not work for large matrices to store t and lambdas, noone knows why.


print "Received transmission matrices of dimensions:"
print "# pixels: %i \n# input angles: %i \n# wavelengths: %i" % np.shape(t)  

N_pix, N_k, N_l = np.shape(t)
n_pix = np.sqrt(N_pix)
pic_shape = (n_pix, n_pix)


# # Calculate which pixels to keep and which are dark ones
# print "Cutting away dark pixels"
# t_amp = np.abs(t)
# int_mean = np.mean(t_amp**2, axis=(1,2))
# av_int_mean = np.mean(int_mean)
# dark_mask = int_mean < 0.15 * av_int_mean


# Calculate all needed ranges etc.
#t = t[:,59:62,:]
#N_pix, N_k, N_l = np.shape(t)

l_range = lambdas[-1] - lambdas[0]
dl = lambdas[1] - lambdas[0]

lambdas_shift = np.arange(N_l) * dl

w = np.linspace(0.0, 2.0*np.pi/dl, N_l)
w_range = w[-1] - w[0]
dw = 2.0 * np.pi / l_range


# perform Fourier transform
print "Performing Fourier transform"
t_ft = np.fft.fftshift(np.fft.fft(t, axis=2), axes=2)

# roll data in Fourier space to compensate for possible linear slope in phase
print "Rolling data in Fourier space"
spec = np.mean(np.abs(t_ft)**2, axis=(0,1))

Norm_spec = np.sum(spec)
spec /= Norm_spec

indx_roll = 0

for i in range(15):
    w_mean = np.einsum('w,w', w, spec)
    indx_roll_inc = int((np.mean(w) - w_mean) / dw)
    indx_roll += indx_roll_inc
    spec = np.roll(spec, indx_roll_inc, axis=0)

t_ft_roll = np.roll(t_ft, indx_roll, axis=2)


print
print "center of mass of average power spectrum: ", w_mean
print "center of wavelength frequencies: ", np.mean(w)
print 


# perform inverse Fourier transform of rolled data
print "Performing inverse Fourier transform of rolled data"
t_roll = np.fft.ifft(np.fft.ifftshift(t_ft_roll, axes=2), axis=2)
np.save(file_mat+'_roll', t_roll)
del t_ft


# separate rolled data in absolute value and phase
print "Separating rolled data in absolute value and phase"
t_amp = np.abs(t_roll)
t_phase = np.unwrap(np.angle(t_roll))


# perform Fourier transform of absolute value and phase
print "Performing Fourier transform of absolute value and phase"
t_amp_ft = np.fft.fftshift(np.fft.fft(t_amp, axis=2), axes=2)
t_phase_ft = np.fft.fftshift(np.fft.fft(t_phase, axis=2), axes=2)
np.save(file_mat+'_amp_ft', t_amp_ft)
np.save(file_mat+'_phase_ft', t_phase_ft)


# calculate average intensity correlation width
print "Calculating average intensity correlation width"
intensity_n = t_amp**2#[dark_mask]
intensity_0 = intensity_n[:,:,0]

mean_int = np.mean(intensity_n, axis=2)
weights = mean_int / np.sum(mean_int, axis=0)

av_int_0_int_n = np.einsum('xk,xkl,xk->kl', intensity_0, intensity_n, weights)
av_int_0 = np.einsum('xk,xk->k', intensity_0, weights)
av_int_n = np.einsum('xkl,xk->kl', intensity_n, weights)

iCorr = np.einsum('kl,k,kl->kl',av_int_0_int_n, 1.0 / av_int_0, 1.0 / av_int_n) - 1
iCorr_dist = np.einsum('kl,k->kl', np.abs(iCorr), 1.0 / np.sum(np.abs(iCorr), axis=1))

plt.plot(lambdas_shift, np.mean(iCorr, axis=0))
plt.savefig(file_mat+'_correlation.png')
plt.clf()

# drop_mask = iCorr <= 0
# iCorr_drop = copy.deepcopy(iCorr)
# iCorr_drop[drop_mask] = 0.0
# iCorr_drop = np.einsum('kl,k->kl', iCorr_drop, 1.0 / np.sum(iCorr_drop, axis=1))
# iCorr_Width_sqr = np.einsum('l,kl->k', lambdas_shift**2, iCorr_drop)

iCorr_Width_sqr = np.einsum('l,kl->k', lambdas_shift**2, iCorr_dist)
iCorr_Width = np.sqrt(iCorr_Width_sqr)
mean_iCorr_Width = np.mean(iCorr_Width)

print "mean intensity correlation width: ", mean_iCorr_Width
del intensity_n, intensity_0


# filter data
print "Filtering data"
filter_params_amp = (15.0 * 2.0 * np.pi / (2.0 * mean_iCorr_Width), w_range/300.0)
filter_params_phase = (15.0 * 2.0 * np.pi / (2.0 * mean_iCorr_Width), w_range/300.0)

print
print ("Ratio of filter width to spectrum (amplitude): ",
       2.0 * filter_params_amp[0] / (w[-1] - w[0]))
print ("Ratio of filter width to spectrum (phase): ",
       2.0 * filter_params_amp[0] / (w[-1] - w[0]))
print

mean_power_amp = np.mean(np.abs(t_amp_ft)**2, axis=(0,1))
N_amp_ft = np.sum(mean_power_amp)

av_amp_ft = np.einsum('w,w', w, mean_power_amp / N_amp_ft)
av_sqr_amp_ft = np.einsum('w,w', w**2, mean_power_amp / N_amp_ft)
spread_amp_ft = np.sqrt(av_sqr_amp_ft - av_amp_ft**2)

t_amp_ft_filter = dpu.low_pass_filter(t_amp_ft,
                                      w - av_amp_ft,
                                      dpu.fermi,
                                      filter_params_amp,
                                      False)


mean_power_phase = np.mean(np.abs(t_phase_ft)**2, axis=(0,1))
N_phase_ft = np.sum(mean_power_phase)

av_phase_ft = np.einsum('w,w', w, mean_power_phase / N_phase_ft)
av_sqr_phase_ft = np.einsum('w,w', w**2, mean_power_phase / N_phase_ft)
spread_phase_ft = np.sqrt(av_sqr_phase_ft - av_phase_ft**2)

t_phase_ft_filter = dpu.low_pass_filter(t_phase_ft,
                                        w - av_phase_ft,
                                        dpu.fermi,
                                        filter_params_phase,
                                        False)


# perform inverse Fourier transform of filtered data
print "Performing inverse Fourier transform of filtered data"
t_amp_filter = dpu.norm_ifft_shift(t_amp, t_amp_ft_filter)
t_phase_filter = dpu.norm_ifft_shift(t_phase, t_phase_ft_filter)
del t_amp, t_phase, t_amp_ft_filter, t_phase_ft_filter

t_filter = t_amp_filter * np.exp(1.0j * t_phase_filter)
np.save(file_mat+'_filter', t_filter)


# check if phase factors correct
phase_factor_direct = np.exp(1.0j * t_phase_filter)
phase_factor_exp = t_filter / np.abs(t_filter)
weighted_phase_diff = (np.sum(np.abs(phase_factor_direct - phase_factor_exp)
                              * np.abs(t_filter)**2) / np.sum(np.abs(t_filter)**2))
print "Weighted difference of phase factors check: ", weighted_phase_diff


# plot for checking
pixel_choice = 50 + n_pix * 50 
k_choice = -1
l_choice = 60

plt.plot(lambdas, np.abs(t_roll[pixel_choice,k_choice,:]), '-b', lw=1.0)
plt.plot(lambdas, np.abs(t_filter[pixel_choice,k_choice,:]), 'd-r', lw=1.0)
plt.savefig(file_mat+'_amp_check.png')
plt.clf()

plt.plot(lambdas, np.unwrap(np.angle(t_roll[pixel_choice,k_choice,:])), '-b', lw=1.0)
plt.plot(lambdas, np.unwrap(np.angle(t_filter[pixel_choice,k_choice,:])), 'd-r', lw=1.0)
#plt.plot(lambdas, t_phase_filter[pixel_choice,k_choice,:].real, '-g', lw=1.0)
#plt.plot(lambdas, np.unwrap(np.angle(t_filter_2[pixel_choice,k_choice,:])), '--k', lw=2.0)
plt.savefig(file_mat+'_phase_check.png')
plt.clf()


sup = np.random.random((N_k)) + 1.0j * np.random.random((N_k))
sup /= np.linalg.norm(sup)

#np.save(file_mat+'_sup', sup)
#sup = np.load(file_mat+'_sup.npy')

out_vec = np.einsum('xkl,k->xl', t, sup)[:,l_choice]
out_vec_roll = np.einsum('xkl,k->xl', t_roll, sup)[:,l_choice]
out_vec_filter = np.einsum('xkl,k->xl', t_filter, sup)[:,l_choice]

pic_shape = (n_pix, n_pix)

out_pic = np.reshape(out_vec, pic_shape)
out_pic_roll = np.reshape(out_vec_roll, pic_shape)
out_pic_filter = np.reshape(out_vec_filter, pic_shape)

gs = gridspec.GridSpec(1, 3, height_ratios=(1,), width_ratios=(1, 1, 1))

plt.subplot(gs[0,0])
plt.pcolor(np.abs(out_pic))
plt.subplot(gs[0,1])
plt.pcolor(np.abs(out_pic_roll))
plt.subplot(gs[0,2])
plt.pcolor(np.abs(out_pic_filter))
plt.savefig(file_mat+'_filter-recovery.png')










