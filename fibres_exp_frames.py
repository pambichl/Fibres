#!/usr/bin/env python

import numpy as np
import subprocess
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec


#file_mat = '0923_126lambda_208k'
file_mat = '0925_101lambda_256k'

file_dir = '../Data-Numerics/' + file_mat + '/'

lambdas = np.load(file_dir + file_mat + '_wavelength.npy')

# Calculate all needed ranges etc.
N_l = np.shape(lambdas)[0]

l_range = lambdas[-1] - lambdas[0]
dl = lambdas[1] - lambdas[0]

lambdas_shift = np.arange(N_l) * dl

w = np.linspace(0.0, 2.0*np.pi/dl, N_l)
w_range = w[-1] - w[0]
dw = 2.0 * np.pi / l_range

### params ###
s_rat = 0.100
cost = 4
CMmax = 1e7
CMmax = None
mov = 1
### ###

file_in = file_dir + "cost=%i/" % cost + file_mat + "_%.3f" % s_rat

if (mov == 1):
    out_vec_PM, out_vec_k, out_vec_opt, out_vec_rand = np.load(file_in + '_vec_output.npy')  
elif (mov == 2):
    out_vec_PM, out_vec_k, out_vec_opt, out_vec_rand = np.load(file_in + '_pack_input.npy')
elif (mov == 3):
    out_vec_PM, out_vec_k, out_vec_opt, out_vec_rand = np.load(file_in + '_pack_output.npy')

N_pix = np.shape(out_vec_PM)[0]
n_pix = np.sqrt(N_pix)
pic_shape = (n_pix, n_pix)

if (not os.path.isdir(file_dir + "cost=%i/movie/s=%.3f/" % (cost, s_rat))):
    subprocess.call(["mkdir", file_dir + "cost=%i/movie/s=%.3f/" % (cost, s_rat)])

for l_indx, l in enumerate(lambdas[0:]):
    print "Producing frame %i of %i" % (l_indx+1,N_l)

    out_pic_PM = np.reshape(np.abs(out_vec_PM[:,l_indx])**2, pic_shape)
    out_pic_k = np.reshape(np.abs(out_vec_k[:,l_indx])**2, pic_shape)
    out_pic_opt = np.reshape(np.abs(out_vec_opt[:,l_indx])**2, pic_shape)
    out_pic_rand = np.reshape(np.abs(out_vec_rand[:,l_indx])**2, pic_shape)

    gs = gridspec.GridSpec(2, 3)#, height_ratios=(1,1,1), width_ratios=(1,1,1))

    plt.xlim(0,n_pix)
    plt.ylim(0,n_pix)
    plt.suptitle(r'$\lambda=$' + '%.4f'%l) 

    plt.subplot(gs[0,0])
    plt.title('random input') 
    plt.pcolor(np.abs(out_pic_rand), vmin=0.0, vmax=CMmax)
    plt.subplot(gs[0,1])
    plt.title('single k') 
    plt.pcolor(np.abs(out_pic_k), vmin=0.0, vmax=CMmax)
    plt.subplot(gs[1,0])
    plt.title('principal mode') 
    plt.pcolor(np.abs(out_pic_PM), vmin=0.0, vmax=CMmax)
    plt.subplot(gs[1,1])
    plt.title('optimized input') 
    plt.pcolor(np.abs(out_pic_opt), vmin=0.0, vmax=CMmax)
    plt.subplot(gs[0:2,2])
    plt.title('wavelength')
    plt.ylim(0.0,1.0)
    plt.vlines(np.mean(lambdas), 0.0, 1.0, lw=2.0, color="black")
    plt.vlines(lambdas[l_indx], 0.0, 1.0, lw=1.0, color="red")
    plt.plot(lambdas, np.zeros((N_l,)))

    plt.savefig(file_dir + "cost=%i/movie/s=%.3f/" % (cost, s_rat) + file_mat + '.s=%.3f.%04i.jpg' % (s_rat, l_indx))
    plt.clf()

    
print "Producing Movie"
pics_name = file_dir + "cost=%i/movie/s=%.3f/" % (cost, s_rat) + file_mat + '.s=%.3f.*.jpg' % s_rat
movie_name =  file_dir + "cost=%i/movie/s=%.3f/" % (cost, s_rat) + "speckles." + file_mat

subprocess.call(["mencoder",
                 "mf://" + pics_name,
                 "-mf", "w=1500:h=600:fps=%f:type=jpg" % (N_l/30.0),
                 "-ovc", "lavc",
                 "-of", "avi",
                 "-o", movie_name + ".avi"])

subprocess.call(["ffmpeg", "-i", movie_name + ".avi", "-s", "1280x960", "-b", "1000k", "-vcodec", "wmv2", "-ar" ,"44100", "-ab", "56000", "-ac", "2", "-y", movie_name + ".wmv"])
