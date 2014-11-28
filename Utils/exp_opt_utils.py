#!/usr/bin/env python

import numpy as np


def calc_phi_gauss(lambdas, lambda_0, t0, s):
   '''
   Calculates Gaussian amplitude function for 
   a packet with center wavelength at lambda_0,
   standard deviation in frequency space 2*s,
   and peak at time t=t0.
   '''
   return (1.0 / (2.0 * np.pi))**(1.0/4.0) * \
       1.0 / np.sqrt(s) * \
       np.exp( -((lambdas - lambda_0) / (2.0 * s))**2 ) * \
       np.exp( 1.0j * t0 * (lambdas - lambda_0) )


def normalize(vec):
   '''Wrapper function for normalization of vector.'''

   vec = vec / np.linalg.norm(vec) 
 
   return vec


def calc_corr_vec(vec, l, indx_0):
    '''
    Calculate spectral intensity correlation function
    for a given output speckle pattern.
    '''
    
    intensity_n = np.abs(vec)**2 / np.sum(np.abs(vec)**2, axis=0)
    intensity_0 = intensity_n[:,indx_0]

    mean_int = np.mean(intensity_n, axis=1)
    weights = mean_int / np.sum(mean_int)
    #weights = mean_int / mean_int

    av_int_0_int_n = np.einsum('x,xl,x->l', intensity_0, intensity_n, weights)
    av_int_0 = np.einsum('x,x', intensity_0, weights)
    av_int_n = np.einsum('xl,x->l', intensity_n, weights)

    iCorr = np.einsum('l,l->l',av_int_0_int_n, 1.0 / av_int_n) / av_int_0 - 1
    iCorr /= iCorr[indx_0]
    #iCorr_dist = np.einsum('kl,k->kl', np.abs(iCorr), 1.0 / np.sum(np.abs(iCorr), axis=1))

    return iCorr


class Functional:
    def __init__(self, phi, t, lambdas, file_out, load_mat):

       N_pix, N_k, N_l = np.shape(t)

        # matrices for longidtudinal spread
        if (load_mat):
            print "Loading cost function matrices"
            self.T0, self.T1, self.T2 = np.load(file_out+'_costfuncmat.npy')

        elif (not load_mat):

            dl = lambdas[1] - lambdas[0]

            print "Calculating transmission matrix x envelope"
            self.phit = np.einsum('xkl,l->xkl', t, phi)
        
            print "Calculating transmission matrix x envelope increment"
            self.dphit = (np.roll(self.phit, 1, axis=2) - np.roll(self.phit, -1, axis=2)) / (2.0 * dl)
            self.dphit[:,:,0] = (self.phit[:,:,1] - self.phit[:,:,0]) / dl
            self.dphit[:,:,-1] = (self.phit[:,:,-1] - self.phit[:,:,-2]) / dl

            print "Calculating cost function matrices"
            t2 = np.einsum('xkl,xql->kql', self.dphit.conj(), self.dphit) * dl
            self.T2 = np.sum(t2, axis=2) * dl
            t1 = (-1.0j) * np.einsum('xkl,xql->kql', self.phit.conj(), self.dphit)
            self.T1 = np.sum(t1, axis=2) * dl
            self.T1 =  0.5 * (self.T1 + self.T1.conj().T) # gives smaller spreads
            t0 = np.einsum('xkl,xql->kq', self.phit.conj(), self.phit) * dl
            self.T0 = np.sum(t0, axis=2)
            np.save(file_out+'_costfuncmat', (self.T0,self.T1,self.T2))

            t0 = np.zeros((N_k,N_k,N_l), dtype='complex')
            t1 = np.zeros((N_k,N_k,N_l), dtype='complex')
            t2 = np.zeros((N_k,N_k,N_l), dtype='complex')
        
            for k in range(N_k):
               for q in range(k):
                  t_k = t[:,k,:]
                  t_q = t[:,q,:]
                  
                  phit_k = np.einsum('xl,l->xl', t_k, phi)
                  phit_q = np.einsum('xl,l->xl', t_q, phi)

                  dphit_k = (np.roll(phit_k, 1, axis=1) - np.roll(phit_k, -1, axis=1)) / (2.0 * dl)
                  dphit[:,0] = (phit[:,1] - phit[:,0]) / dl
                  dphit[:,-1] = (phit[:,-1] - phit[:,-2]) / dl
                  
                  t0[k,q,:] = 



    def calc_coeffs(self, phat):
        '''
        Calculates coefficients for eigenvalue
        equation, i.e., the expectation values
        of the matrices A, C, D with vector phat.
        '''
        
        a = np.dot( phat.conj(), np.dot( self.T2, phat))
        c = np.dot( phat.conj(), np.dot( self.T1, phat))
        d = np.dot( phat.conj(), np.dot( self.T0, phat))
            
        return a, c, d
   
       
    def G_long(self, phat):
        '''
        Apply non-linear gradient function for longitudinal spread
        to vector phat.
        '''

        a, c, d = self.calc_coeffs(phat)
    
        G = (-a/d**2 + 2.0*c**2/d**3) * np.dot(self.T0, phat) +\
        1.0/d * np.dot(self.T2, phat) -\
        2.0*c/d**2 * np.dot(self.T1, phat)
  
        return G 


    def L_long(self, phat):
        '''
        Functional to be minimized giving longitudinal spread.
        '''

        a, c, d = self.calc_coeffs(phat)
    
        L = a/d - (c/d)**2
    
        return L
