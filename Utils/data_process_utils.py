#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.io

import data_process_utils as dpu

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def read_write_stack(name, data=None, write=0):
    '''
    Utility function writing or loading data to or from file.
    '''
    if (write):
        np.save(name, data)
    elif (not write):
        data = np.load(name+'.npy')
    
    return data


def gauss(x, params):
    '''Gaussian to be used as filter function.'''
    s = params[0]
    return np.exp(-0.5 * x**2 / s**2)


def fermi(x, params):
    '''Fermi functions to be used as rectangular filter function.'''
    s, a = params[:2]
    fermi_step_left = 1.0 / (1.0 + np.exp(-(x + s)/ a))
    fermi_step_right = 1.0 / (1.0 + np.exp(-(x - s)/ a))
    return fermi_step_left - fermi_step_right 


def low_pass_filter(signal, wl_vals, filter_func, filter_params, out_filt=False):
    '''
    Applies filter of given shape to signal.
    '''
    Nwl = np.shape(wl_vals)[0]

    def filter_func_wrapper(x):
        '''
        Needed to vectorize function of more than one parameter.
        '''
        return filter_func(x, filter_params)
    
    filter_func_vec = np.vectorize(filter_func_wrapper)(wl_vals)
    #indx_center = np.argmin(np.abs(wl_vals - np.mean(wl_vals)))
    #filter_func_vec /= filter_func_vec[indx_center]
    #filter functions equal to 1 at center anyway
    
    signal_filter = np.einsum('w,xkw->xkw', filter_func_vec, signal)
    
    if (out_filt): 
        return signal_filter, filter_func_vec
    if (not out_filt):
        return signal_filter

    
def norm_ifft(signal, signal_ft):
    '''
    Performs inverse Fourier transform and normalizes
    backtransformed signal (bt_signal) such that it has the same
    mean value as original signal (signal).
    '''
    
    bt_signal = np.fft.ifft(signal_ft, axis=2)
    #mean_signal = np.mean(np.abs(signal), axis=2)
    #mean_bt_signal = np.mean(np.abs(bt_signal), axis=2)
    #bt_signal = np.einsum('xk,xk,xkl->xkl', mean_signal, 1.0/mean_bt_signal, bt_signal)
    
    return bt_signal 


def norm_ifft_shift(signal, signal_ft):
    '''
    Performs inverse Fourier transform including back shift and normalizes
    backtransformed signal (bt_signal) such that it has the same
    mean value as original signal (signal).
    '''
    
    bt_signal = np.fft.ifft(np.fft.ifftshift(signal_ft, axes=2), axis=2)
    #mean_signal = np.mean(np.abs(signal), axis=2)
    #mean_bt_signal = np.mean(np.abs(bt_signal), axis=2)
    ### global phase of each pixel messed up somehow!!!
    #bt_signal = np.einsum('xk,xk,xkl->xkl', mean_signal, 1.0/mean_bt_signal, bt_signal)
    
    return bt_signal
    

def plot_ft(**data):
    '''
    Produces output related to Fourier transforms
    of Wen's transmission matrix data.
    '''

    for param, value in data.iteritems():
        globals()[param] = value

    plt.plot(lambdas, pixel_real, 'd-b', lw=0.5)
    plt.plot(lambdas, pixel_filter_real, '-r', lw=2.5)
    plt.plot(lambdas, pixel_imag, 'd-g', lw=0.5)
    plt.plot(lambdas, pixel_filter_imag, '-o', lw=2.5)
    plt.title("real part (lambda)")
    plt.savefig("real.png")
    plt.clf()
    
    plt.plot(wlambdas, pixel_ft_abs, '-b', lw=0.5)
    plt.plot(wlambdas, mean_t_ft_abs, '-g', lw=0.5)
    #plt.plot(wlambdas, filter_func, '-r', lw=2.5)
    plt.title("abs (w)")
    plt.savefig("ft_abs.png")
    plt.clf()

    plt.plot(lambdas, pixel_abs, 'd-b', lw=0.5)
    plt.title("abs (lambda)")
    plt.savefig("abs.png")
    plt.clf()

    plt.plot(lambdas, pixel_phase, '-b', lw=2.5)
    #plt.plot(lambdas, pixel_line, '-g', lw=2.5)
    plt.title("phase (lambda)")
    plt.savefig("phase.png")
    plt.clf()
    




