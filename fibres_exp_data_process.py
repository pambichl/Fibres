#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.io

import data_process_utils as dpu

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


# read transmission matrices
file_mat = '0917_401lambda_1k'
#file_mat = '101_wavelength'

print "Reading in transmission matrices"
try:
    t, lambdas = np.load(file_mat+'.npy')
except:
    data = scipy.io.loadmat('../Wen/'+file_mat+'.mat')
    t = data['TM']
    lambdas = data['wavelength'][0,:]
    np.save(file_mat, (t, lambdas))

print "Received transmission matrices of dimensions:"
print "# pixels: %i \n# input angles: %i \n# wavelengths: %i" % np.shape(t)  
 

Npixel = np.shape(t)[0]
Npixel_xy = np.sqrt(Npixel)

Nlambda = np.shape(lambdas)[0]
lambda_range = lambdas[-1] - lambdas[0]
dlambda = lambdas[1] - lambdas[0]

wlambdas = np.linspace(0.0, 2.0*np.pi/dlambda, Nlambda)
dwlambda = 2.0 * np.pi / lambda_range


print "Performing Fourier transform of transmission matrices"
#t_ft = np.fft.fftshift(np.fft.fft(t, axis=2), axes=2)
#dpu.read_write_stack('t_ft', t_ft, 1)
t_ft = dpu.read_write_stack('t_ft')

t_ft_abs = np.abs(t_ft)
mean_t_ft_abs = np.mean(t_ft_abs[:,0,:], axis=0)

# plt.plot(wlambdas, t_ft_abs[500,0,:])
# plt.show()

t_pic = np.reshape(t, (Npixel_xy, Npixel_xy, 1, Nlambda))

# plt.pcolor(np.abs(t_pic[:,:,0,0]))
# plt.show()
# plt.clf()



t_ft_filter, filter_func = dpu.low_pass_filter(t_ft,
                                              wlambdas,
                                              filter_func=dpu.fermi,
                                              sigma=7.5, #* dwlambda,
                                              out_filt=1)


print "Performing inverse Fourier transform of transmission matrices"
#t_filter = dpu.norm_ifft(t, t_ft_filter)
#dpu.read_write_stack('t_filter', t_filter, 1)
t_filter = dpu.read_write_stack('t_filter')


pixel = 204 + Npixel_xy * 182

pixel_real = np.real(t[pixel,0,:])
pixel_imag = np.imag(t[pixel,0,:])
pixel_abs = np.abs(t[pixel,0,:])
pixel_phase = np.unwrap(np.angle(t[pixel,0,:]))
pixel_slope = (pixel_phase[-1] - pixel_phase[0]) / (lambdas[-1] - lambdas[0])
pixel_line = pixel_slope * (lambdas - lambdas[0]) + pixel_phase[0]

pixel_phase -= pixel_line

#pixel_phase = np.angle(t[pixel,0,:])

pixel_filter_real = np.real(t_filter[pixel,0,:])
pixel_filter_imag = np.imag(t_filter[pixel,0,:])
pixel_ft_abs = np.abs(t_ft[pixel,0,:])


print "Producing output"
dpu.plot_ft(**globals())


### intensity correlation function ###
# add average over different lambdas, atm just for lambda_min
intensity = np.abs(t)**2

i0 = intensity[:,0,0]
il = intensity[:,0,:]

i0_il = np.einsum('x,xl->xl', i0, il)
av_i0 = np.mean(i0)
av_il = np.mean(il, axis=0)
av_i0_il = np.mean(i0_il, axis=0)

iCorr = av_i0_il / (av_i0 * av_il) - 1
iCorr = iCorr / np.max(iCorr)

plt.plot(np.arange(Nlambda)*dlambda, iCorr)
plt.savefig("IntensityCorrelation.png")
plt.clf()






