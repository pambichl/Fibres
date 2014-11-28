#!/usr/bin/env python

import numpy as np
import sys

import copy


def calc_phi_gauss(lambdas, lambda_0, t0, s):
   '''
   Calculates Gaussian amplitude function for 
   a packet with center wavelength at lambda_0,
   standard deviation in frequency space 2*s,
   and peak at time t=t0.
   '''
   return np.array(1.0 / (2.0 * np.pi))**(1.0/4.0) * \
       1.0 / np.sqrt(s) * \
       np.exp( -((lambdas - lambda_0) / (2.0 * s))**2.0 ) * \
       np.exp( 1.0j * t0 * (lambdas - lambda_0) )

def calc_phi_fermi(x, params):
    '''Fermi functions to be used as rectangular filter function.'''
    s, a = params[:2]
    fermi_step_left = 1.0 / (1.0 + np.exp(-(x + 0.5 * s)/ a))
    fermi_step_right = 1.0 / (1.0 + np.exp(-(x - 0.5 * s)/ a))
    return fermi_step_left - fermi_step_right 


def normalize(vec):
   '''Normalizes vector.'''

   vec = vec / np.linalg.norm(vec) 
 
   return vec


def calc_corr_vec(vec, indx_0=None, weights=None):
    '''
    Calculate spectral intensity correlation function
    for a given output speckle pattern.
    '''

    N_x, N_l = np.shape(vec)

    if (indx_0 == None):
       indx_0 = int(np.floor(N_l / 2))
    # normalize intensity profile at each lambda
    intensity_n = np.abs(vec)**2# / np.sum(np.abs(vec)**2, axis=0)
    intensity_0 = intensity_n[:,indx_0]

    if (weights == None):
        #mean_int = np.mean(intensity_n, axis=1)
        #weights = mean_int / np.sum(mean_int)
        #weights = mean_int / mean_int
        weights = np.ones((N_x,))
        weights /= np.sum(weights)
        #weights = intensity_0 / np.sum(intensity_0)

    av_int_0_int_n = np.einsum('x,xl,x->l', intensity_0, intensity_n, weights)
    av_int_0 = np.einsum('x,x', intensity_0, weights)
    av_int_n = np.einsum('xl,x->l', intensity_n, weights)

    iCorr = np.einsum('l,l->l',av_int_0_int_n, 1.0 / av_int_n) / av_int_0 - 1
    iCorr /= iCorr[indx_0]

    return iCorr


# def calc_corr_vec(vec, lambdas, indx_0, weights=None):
#     '''
#     Calculate spectral intensity correlation function
#     for a given output speckle pattern.
#     '''

#     N_l = np.shape(lambdas)[0]
    
#     # normalize intensity profile at each lambda
#     I_n = np.abs(vec)**2# / np.sum(np.abs(vec)**2, axis=0)

#     if (weights == None):
#         mean_int = np.mean(intensity_n, axis=1)
#         weights = mean_int / np.sum(mean_int)
        
#     iCorr = np.zeros((N_l,), dtype='complex')
#     for l in range(N_l):
#          iCorr[l] = np.einsum('xl,xl,x', I_n, np.roll(I_n, -l, axis=1), weights)

#     iCorr = np.fft.fftshift(iCorr)
#     iCorr /= iCorr[indx_0]

#     return iCorr


def calc_corr_field(vec, lambdas, indx_0, weights=None, env=False):
    '''
    Calculate spectral intensity correlation function
    for a given output speckle pattern.
    '''

    N_l = np.shape(lambdas)[0]
    
    vec_n = vec
    if (np.abs(env).any()):
        vec_n = np.einsum('xl,l->xl', vec, env)
    
    if (weights == None):
        intensity_n = np.abs(vec_n)**2
        mean_int = np.mean(intensity_n, axis=1)
        weights = mean_int / np.sum(mean_int)

    vecCorr = np.zeros((N_l,), dtype='complex')
    vecCorrEnv = np.zeros((N_l,), dtype='complex')
    for l in range(N_l):
        vecCorr[l] = np.einsum('xl,xl,x', vec_n.conj(), np.roll(vec_n, -l, axis=1), weights)

    vecCorr = np.fft.fftshift(vecCorr)
    vecCorr /= vecCorr[indx_0]
    
    return vecCorr


def optimize(L, G, phat_old, ds0, nloop):
    '''
    Numerical minimization of functional L with gradient G.
    '''
    L_old = L(phat_old) 
    L_min = L_old
    i_min = 0
    phat_min = phat_old
    print "step: %5i of %5i gradient: %4.6f %4.6f\r" % (0, nloop, L_old.real, np.linalg.norm(G(phat_old))),
    for i in range(nloop):
        
        # calculate what value functional would have if step would be made
        phat_old = normalize(phat_old) 
        L_old = L(phat_old)   
        G_old = G(phat_old)   
        ds = ds0 
        phat_int = phat_old - G(phat_old) * ds
        phat_int = normalize(phat_int) 
        L_int = L(phat_int) 
        G_int = G(phat_int)   
      
        # if new value is larger than old one reduce step size until it is smaller
        count = 0
        while ((L_old < L_int and count <= 30)): #or
            #abs((L_old - L_int) / L_old) > 0.00001 and
            #(np.linalg.norm(G_old) < np.linalg.norm(G_int) and count <= 500)):
            ds = 0.50*ds
            phat_int = phat_old - G(phat_old) * ds
            phat_int = normalize(phat_int) 
            L_int = L(phat_int)
            G_int = G(phat_int)   
            AbsGrad = np.linalg.norm(G(phat_int))
            count += 1

        # finally perform step with reduced step size
        phat_new = phat_old - G(phat_old) * ds
        phat_new = normalize(phat_new) 
        L_new = L(phat_new)
        G_new = G(phat_new)   

        if (L_new < L_min):
            L_min = L_new
            G_min = G_new
            phat_min = phat_new
            i_min = i

        AbsGrad = np.linalg.norm(G(phat_new))
        if (i % np.floor(nloop/1000.0) == 0):
           if (count == 0):
              print "\rstep: %5i of %5i gradient: %4.10f %4.10f" % (i+1, nloop, L_new.real, AbsGrad), 
           else:
              print "\rstep: %5i of %5i gradient: %4.10f %4.10f step %i" % (i+1, nloop, L_new.real, AbsGrad, count),
           sys.stdout.flush()

        if (count >= 30): 
            print "\nStep size break"
            break
        phat_old = copy.deepcopy(phat_new)   

    print
    print "Found minimum of functional and gradient at step %i: %4.6f, %4.6f" % (i_min+1, L_min.real, np.linalg.norm(G(phat_min)))
    phat = phat_min

    return normalize(phat)


class Functional:
    def __init__(self, phi, t, lambdas, lambda_0, weights, file_out, load_mat, cost, sprex=2):
        '''
        Class containing all prerequisites to calculate functional to be minimized
        and its gradient function.

        cost = 1: calculate variance in Fourier-space directly from lambda-space.
        cost = 2: calculate absolute difference to mean output to the power of
                  sprex in Fourier-space of lambda.
        cost = 3: calculate variance in Fourier-space in Fourier space of lambda.
        cost = 4: calculate absolute difference to mean output to the power of
                  1 in Fourier space of lambda.
        '''

        file_dir = "/".join(file_out.split("/")[:3]) + "/"
        file_mat = file_out.split("/")[2]
        
        N_pix, N_k, N_l = np.shape(t)

        dl = lambdas[1] - lambdas[0]
        l_range = lambdas[-1] - lambdas[0]
        
        dl_FT = 2.0 * np.pi / l_range
        lambdas_FT = np.arange(0,N_l) * dl_FT
        self.lambdas_FT = lambdas_FT
        self.dl_FT = dl_FT

        l_vals = np.arange(5,N_l-5)
        l_vals = np.arange(0,N_l)
        # omit first and last lambdas because not reliable due to filtering
        lambdas_rel = lambdas / lambda_0
        dl_rel = dl / lambda_0
        # lambdas relative to reference wavelength

        indx_0 = np.argmin(np.abs(lambdas-lambda_0))

        self.sprex = sprex
        
        # matrices for longidtudinal spread
        if (load_mat):
            print "Loading cost function matrices"
            if (cost==1):
                self.T0, self.T1, self.T2 = np.load(file_out+'_costfuncmat.npy')
            if (cost==2):
                self.T0, self.T1 = np.load(file_out+'_costfuncmat.npy')
                self.Tl = np.load(file_out+'_tempIntmat.npy')
            if (cost==3):
                self.T0, self.T1, self.T2 = np.load(file_out+'_costfuncmat.npy')
            if (cost==4):
                self.T0, self.T1 = np.load(file_out+'_costfuncmat.npy')

        elif (not load_mat):
            u, v = np.load(file_dir + file_mat + '_singmats.npy')
            N_x = np.shape(t)[0] 
            N_k = np.shape(t)[1] 
            N_f = np.shape(u)[1]

            np.save(file_out+'_check1',t)

            print "Projecting transmission matrices"
            for l in range(N_l):
                print "\rlambda %i of %i" % (l+1, N_l),
                sys.stdout.flush()

                proj = u.conj().T.dot(t[:,:,l]).dot(v)
                t[:,:,l] = u.dot(proj).dot(v.conj().T)
                # CAUTION: t matrix overwritten here

            #np.save(file_out+'_projection', t)
            #t = np.load(file_out+'_projection.npy')
            # CAUTION: t matrix overwritten here
            
            np.save(file_out+'_check2',t)    

            print
            print "Calculating cost function matrices"
            
            T0_l = np.zeros((N_k,N_k,N_l), dtype='complex')
            T1_l = np.zeros((N_k,N_k,N_l), dtype='complex')
            T2_l = np.zeros((N_k,N_k,N_l), dtype='complex')
            
            if (cost == 1):
                # print "ZNADA"
                # phi_step = np.hstack(((phi[indx_0::-3])[::-1], (phi[indx_0::3])[1:]))
                # print phi_step
                # phi_full = np.vstack((phi_step, phi_step, phi_step)).T.flatten()
                # print phi_full
                # phi = phi_full
                for l in l_vals:
                #for l in (indx_0-6,indx_0-3,indx_0,indx_0+3,indx_0+6): 
                #for l in range(N_l):
                    print "\rlambda %i of %i" % (l-l_vals[0]+1, len(l_vals)),
                    sys.stdout.flush()
                    
                    # super PM:
                    # Rechenstellen: (indx_0-3,indx_0,indx_0+3)
                    # Start: PM0
                    # phi: konstant 1

                    #phi = np.ones(np.shape(phi))
                    #phit = np.einsum('xkl,l->xkl', t[:,:,l-1:l+2], phi[l-1:l+2])
                    
                    #dl_dw = -lambdas[l]**2 / (2.0 * np.pi) / lambdas[0]**2
                    #dl_dw = 1

                    if l == 0:
                        #dphit = (t[:,:,1] * phi[1] - t[:,:,0] * phi[0]) / dl
                        dphit = ((t[:,:,1] - t[:,:,0]) * phi[0] + 
                                 (phi[1] - phi[0]) * t[:,:,0]) / dl

                    elif l == N_l-1:
                        #dphit = (t[:,:,-2] * phi[-2] - t[:,:,-1] * phi[-1]) / dl
                        dphit = ((t[:,:,-2] - t[:,:,-1]) * phi[-1] + 
                                 (phi[-2] - phi[-1]) * t[:,:,-1]) / dl

                    else:
                        #dphit = (t[:,:,l+1] * phi[l+1] - t[:,:,l-1] * phi[l-1]) / (2.0 * dl)
                        dphit = ((t[:,:,l+1] - t[:,:,l-1]) * phi[l] + 
                                 (phi[l+1] - phi[l-1]) * t[:,:,l]) / (2.0 * dl)

                    #phit = phit[:,:,1]
                    phit = t[:,:,l] * phi[l]
                        
                    T0_l[:,:,l] = np.einsum('xk,x,xq->kq',
                                            phit.conj(), weights, phit) * dl
                    T1_l[:,:,l] = (-1.0j) * np.einsum('xk,x,xq->kq',
                                                      phit.conj(), weights, dphit)  * dl
                    T2_l[:,:,l] = np.einsum('xk,x,xq->kq',
                                            dphit.conj(), weights, dphit) * dl

                self.T0 = np.sum(T0_l, axis=2) 
                self.T1 = np.sum(T1_l, axis=2)
                self.T2 = np.sum(T2_l, axis=2)
                self.T1 = 0.5 * (self.T1 + self.T1.conj().T)
                print
                print "Saving cost function matrices"
                np.save(file_out+'_costfuncmat', (self.T0,self.T1,self.T2))
                
            if (cost == 2):
                N_x = np.shape(t)[0] 
                N_k = np.shape(t)[1] 

                print "Calculating Fourier transform"
                for k in range(N_k):
                    print "\rangle %i of %i" % (k+1, N_k),
                    sys.stdout.flush()
                    
                    phit_k = np.einsum('xl,l->xl', t[:,k,:], phi)
                    t[:,k,:] = np.fft.fftshift(np.fft.fft(phit_k, axis=-1), axes=-1)
                    # CAUTION: t matrix overwritten here

                print
                print "Calculating power matrix"
                T_FT = np.zeros((N_k,N_k,N_l), dtype='complex')
                for l in range(N_l):
                    print "\rlambda %i of %i" % (l+1, N_l),
                    sys.stdout.flush()

                    T_FT[:,:,l] = np.einsum('xk,xq->kq', t[:,:,l].conj(), t[:,:,l])

                self.T0 = np.einsum('kql,l->kq', T_FT, lambdas_FT**0)
                self.T1 = np.einsum('kql,l->kq', T_FT, lambdas_FT**1)
                self.Tl = T_FT
                print
                print "Saving cost function matrices"
                np.save(file_out+'_costfuncmat', (self.T0,self.T1))
                np.save(file_out+'_tempIntmat', self.Tl)

            if (cost == 3):
                N_x = np.shape(t)[0] 
                N_k = np.shape(t)[1] 

                np.save(file_out+'_check3',t)    

                print "Calculating Fourier transform"
                phit_store = np.zeros(np.shape(t), dtype="complex")
                for k in range(N_k):
                    print "\rangle %i of %i" % (k+1, N_k),
                    sys.stdout.flush()
                    
                    phit_k = np.einsum('xl,l->xl', t[:,k,:], phi)
                    phit_store[:,k,:] = phit_k
                    t[:,k,:] = np.fft.fftshift(np.fft.fft(phit_k, axis=-1), axes=-1)
                    # CAUTION: t matrix overwritten here

                np.save(file_out+'_check4',t)   
                np.save(file_out+'_check5',phi)   
                np.save(file_out+'_check6',phit_store)  

                print
                print "Calculating power matrix"
                T_FT = np.zeros((N_k,N_k,N_l), dtype='complex')
                for l in range(N_l):
                    print "\rlambda %i of %i" % (l+1, N_l),
                    sys.stdout.flush()

                    T_FT[:,:,l] = np.einsum('xk,xq->kq', t[:,:,l].conj(), t[:,:,l])

                self.T0 = np.einsum('kql,l->kq', T_FT, self.lambdas_FT**0)
                self.T1 = np.einsum('kql,l->kq', T_FT, self.lambdas_FT**1)
                self.T2 = np.einsum('kql,l->kq', T_FT, self.lambdas_FT**2)
                print
                print "Saving cost function matrices"
                np.save(file_out+'_costfuncmat', (self.T0,self.T1,self.T2))
                
            if (cost == 4):
                N_x = np.shape(t)[0] 
                N_k = np.shape(t)[1] 

                print "Calculating Fourier transform"
                for k in range(N_k):
                    print "\rangle %i of %i" % (k+1, N_k),
                    sys.stdout.flush()
                    
                    phit_k = np.einsum('xl,l->xl', t[:,k,:], phi)
                    t[:,k,:] = np.fft.fftshift(np.fft.fft(phit_k, axis=-1), axes=-1)
                    # CAUTION: t matrix overwritten here

                print
                print "Calculating power matrix"
                T_FT_Int_0 = np.zeros((N_k,N_k,N_l), dtype='complex')
                T_FT_Int_1 = np.zeros((N_k,N_k,N_l), dtype='complex')
                T_FT_Carr_0 = np.zeros((N_k,N_k), dtype='complex')
                T_FT_Carr_1 = np.zeros((N_k,N_k), dtype='complex')
                PreCarr = np.zeros((N_k,N_k), dtype='complex')

                for l in range(N_l):
                    print "\rlambda %i of %i" % (l+1, N_l),
                    sys.stdout.flush()

                    PreCarr =+ np.einsum('xk,xq->kq', t[:,:,l].conj(), t[:,:,l]) 
                    T_FT_Carr_0 += PreCarr * lambdas_FT[l]**0 
                    T_FT_Carr_1 += PreCarr * lambdas_FT[l]**1 
                    T_FT_Int_0[:,:,l] = T_FT_Carr_0
                    T_FT_Int_1[:,:,l] = T_FT_Carr_1
                    
                self.T0 = T_FT_Int_0
                self.T1 = T_FT_Int_1
                print
                print "Saving cost function matrices"
                np.save(file_out+'_costfuncmat', (self.T0,self.T1))

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
    
    def G_trans(self, phat):
        '''
        Apply non-linear gradient function for transverse spread
        to vector phat.
        '''

        t0 = phat.conj().dot(self.T0).dot(phat)
        t1 = phat.conj().dot(self.T1).dot(phat)
        tl = np.einsum('k,kql,q->l', phat.conj(), self.Tl, phat)

        t_mean = t1 / t0
        
        # C1 = np.einsum('l,l,kql->kq',
        #                np.sign(self.lambdas_FT - t_mean)**self.sprex,
        #                (self.lambdas_FT - t_mean)**self.sprex,
        #                self.Tl) / t0
        # C2 = self.sprex * (np.einsum('l,l,l',
        #                              np.sign(self.lambdas_FT - t_mean)**self.sprex,
        #                              (self.lambdas_FT - t_mean)**(self.sprex-1),
        #                              tl) / t0 * (self.T1 / t0 - t1 / t0**2 * self.T0)
        #                              )
        # C3 = np.einsum('l,l,l',
        #                np.sign(self.lambdas_FT - t_mean)**self.sprex,
        #                (self.lambdas_FT - t_mean)**self.sprex,
        #                tl) / t0**2 * self.T0

        # G = (C1 - C2 - C3).dot(phat)
        
        T0phat = np.einsum('kq,q->k', self.T0, phat)
        T1phat = np.einsum('kq,q->k', self.T1, phat)
        Tlphat = np.einsum('kql,q->kl', self.Tl, phat)
        
        C1 = np.einsum('l,kl->k',
                       (np.sign(self.lambdas_FT - t_mean) * 
                        (self.lambdas_FT - t_mean))**self.sprex,
                       Tlphat) / t0
        C2 = np.sum((np.sign(self.lambdas_FT - t_mean) *
                     (self.lambdas_FT - t_mean))**self.sprex /
                     (self.lambdas_FT - t_mean) * 
                     tl) / t0 * (T1phat / t0 - t1 / t0**2 * T0phat)
                                     
        C3 = np.sum((np.sign(self.lambdas_FT - t_mean) *
                     (self.lambdas_FT - t_mean))**self.sprex *
                     tl / t0**2) * T0phat

        G = (C1 - C2 - C3)
        
        return G 

    def L_trans(self, phat):
        '''
        Functional to be minimized giving transverse spread.
        '''

        t0 = phat.conj().dot(self.T0).dot(phat)
        t1 = phat.conj().dot(self.T1).dot(phat)
        tl = np.einsum('k,kql,q->l', phat.conj(), self.Tl, phat)

        t_mean = t1 / t0
        
        # L = np.einsum('l,l,l',
        #               tl,
        #               np.sign(self.lambdas_FT - t_mean)**self.sprex,
        #               (self.lambdas_FT - t_mean)**self.sprex) / t0

        L = np.sum(tl * 
                   (np.sign(self.lambdas_FT - t_mean) *
                   (self.lambdas_FT - t_mean))**self.sprex) / t0
    
        return L

    def G_long_FT(self, phat):
        '''
        Apply non-linear gradient function for longitudinal spread
        to vector phat.
        '''

        a, c, d = self.calc_coeffs(phat)
    
        G = ((-a/d**2 + 2.0*c**2/d**3) * np.dot(self.T0, phat)
             +1.0/d * np.dot(self.T2, phat)
             -2.0*c/d**2 * np.dot(self.T1, phat))
    
        return G

    def L_long_FT(self, phat):
        '''
        Functional to be minimized giving longitudinal spread.
        '''

        a, c, d = self.calc_coeffs(phat)
    
        L = a/d - (c/d)**2
    
        return L

    def G_long_Abs(self, phat):
        '''
        Apply non-linear gradient function for longitudinal spread
        defined as mean absolute distance to mean value to vector phat.
        Idea is to split integrals over sign functions into below and
        above mean value.
        '''

        t0 = phat.conj().dot(self.T0[:,:,-1]).dot(phat)
        t1 = phat.conj().dot(self.T1[:,:,-1]).dot(phat)

        T0 = self.T0[:,:,-1]
        T1 = self.T1[:,:,-1]
        
        t_mean = np.abs(t1 / t0)
        
        sign_indx = 0.5 * (np.shape(self.lambdas_FT)[0]
                           - np.sum(np.sign(self.lambdas_FT - t_mean))) - 1

        t0_b = -1.0 * phat.conj().dot(self.T0[:,:,sign_indx]).dot(phat)
        t0_a = t0 + t0_b
        t1_b = -1.0 * phat.conj().dot(self.T1[:,:,sign_indx]).dot(phat)
        t1_a = t1 + t1_b 

        T0_b = -1.0 * self.T0[:,:,sign_indx]
        T0_a = self.T0[:,:,-1] + T0_b
        T1_b = -1.0 * self.T1[:,:,sign_indx]
        T1_a = self.T1[:,:,-1] + T1_b
        
        G1 = (1.0 / t0**2
              * (t1_b + t1_a - t_mean * (t0_b + t0_a))
              * self.T0[:,:,-1].dot(phat))
        G2 = (1.0 / t0
              * (T1_b + T1_a - t_mean * (T0_b + T0_a)).dot(phat))
        G3 = (1.0 / t0**3
              * (t0_b + t0_a)
              * t1
              * T0.dot(phat))
        G4 = (1.0 / t0**2
              * (t0_b + t0_a)
              * T1.dot(phat))

        G = -G1 + G2 + G3 - G4
    
        return G

    def L_long_Abs(self, phat):
        '''
        Functional to be minimized giving longitudinal spread, defined
        as mean absolute distance to mean value. Idea is to split
        integrals over sign functions into below and above mean value.
        '''
        
        t0 = phat.conj().dot(self.T0[:,:,-1]).dot(phat)
        t1 = phat.conj().dot(self.T1[:,:,-1]).dot(phat)

        t_mean = np.abs(t1 / t0)
        
        sign_indx = 0.5 * (np.shape(self.lambdas_FT)[0]
                               - np.sum(np.sign(self.lambdas_FT - t_mean))) - 1

        t0_b = -1.0 * phat.conj().dot(self.T0[:,:,sign_indx]).dot(phat)
        t0_a = t0 + t0_b
        t1_b = -1.0 * phat.conj().dot(self.T1[:,:,sign_indx]).dot(phat)
        t1_a = t1 + t1_b

        L = 1.0 / t0 * (t1_b + t1_a - t_mean * (t0_b + t0_a))
        
        return L


def calc_long_spread(vec, lambdas):
    dl = lambdas[1] - lambdas[0]
    #dl_dw = -lambdas**2 / (2.0 * np.pi) / lambdas[0]**2
    dl_dw = 1.0 * np.ones(np.shape(lambdas))

    dvec = (np.roll(vec, -1, axis=1) - np.roll(vec, 1, axis=1)) / (2.0 * dl)
    dvec[:,0] = (vec[:,1] - vec[:,0]) / dl
    dvec[:,-1] = (vec[:,-1] - vec[:,-2]) / dl
    dvec = np.einsum('xl,l->xl', dvec, dl_dw)
        
    t0 = np.einsum('xl,xl,l', vec.conj(), vec, dl / dl_dw) 
    t1 = np.einsum('xl,xl,l', vec.conj(), dvec, dl / dl_dw).real
    t2 = np.einsum('xl,xl,l', dvec.conj(), dvec, dl / dl_dw)

    sigma = np.sqrt(t2 / t0 - (t1 / t0)**2)
    
    return sigma


def calc_trans_spread(vec, lambdas):
    dl = lambdas[1] - lambdas[0]

    t0 = np.einsum('xl,xl,l', vec.conj(), vec, lambdas**0)
    t1 = np.einsum('xl,xl,l', vec.conj(), vec, lambdas**1)
    t2 = np.einsum('xl,xl,l', vec.conj(), vec, lambdas**2)
    
    sigma = np.sqrt(t2 / t0 - (t1 / t0)**2)
    
    return sigma


def calc_long_FT_spread(vec, weights, lambdas):
    vec_FT = np.fft.fftshift(np.fft.fft(vec, axis=-1), axes=-1)

    I = np.einsum('xl,x,xl->l', vec_FT.conj(), weights, vec_FT)
    
    l_range = lambdas[-1] - lambdas[0]
    N_l = np.shape(lambdas)[0]
    
    dl_FT = 2.0 * np.pi / l_range
    lambdas_FT = np.arange(0,N_l) * dl_FT

    t0 = np.einsum('l,l', I**2, lambdas_FT**0) * dl_FT
    t1 = np.einsum('l,l', I**2, lambdas_FT**1) * dl_FT
    t2 = np.einsum('l,l', I**2, lambdas_FT**2) * dl_FT
    
    sigma = np.sqrt(t2 / t0 - (t1 / t0)**2)
    
    return sigma

# passt ueberhaupt nicht mit cost function zusammen!!


# t0 = np.zeros((N_k,N_k), dtype='complex')
# t1 = np.zeros((N_k,N_k), dtype='complex')
# t2 = np.zeros((N_k,N_k), dtype='complex')

# for k in range(N_k):
#    print "row %i of %i" % (k+1, N_k)
#    t_k = t[:,k,:] 
#    phit_k = np.einsum('xl,l->xl', t_k, phi)
#    dphit_k = (np.roll(phit_k, 1, axis=1) - np.roll(phit_k, -1, axis=1)) / (2.0 * dl)
#    dphit_k[:,0] = (phit_k[:,1] - phit_k[:,0]) / dl
#    dphit_k[:,-1] = (phit_k[:,-1] - phit_k[:,-2]) / dl

#    for q in np.arange(k,N_k):
#       t_q = t[:,q,:]              
#       phit_q = np.einsum('xl,l->xl', t_q, phi)
#       dphit_q = (np.roll(phit_q, 1, axis=1) - np.roll(phit_q, -1, axis=1)) / (2.0 * dl)
#       dphit_q[:,0] = (phit_q[:,1] - phit_q[:,0]) / dl
#       dphit_q[:,-1] = (phit_q[:,-1] - phit_q[:,-2]) / dl

#       t0[k,q] = np.einsum('xl,xl', phit_k.conj(), phit_q) * dl  
#       t1[k,q] = np.einsum('xl,xl', phit_k.conj(), dphit_q) * -1.0j * dl
#       t2[k,q] = np.einsum('xl,xl', dphit_k.conj(), dphit_q) * dl



# self.T_FT_l = np.zeros((N_k, N_k, N_l), dtype="complex")
# for l in l_vals:
#     print "\rlambda %i of %i" % (l+1, N_l),
#     sys.stdout.flush()
    
#     self.T_FT_l[:,:,l] = np.einsum('xk,x,xq->kq',
#                                    t[:,:,l].conj(), weights, t[:,:,l])
#     print
#     print "Saving cost function matrices"
#     np.save(file_out+'_costfuncmat', (self.T_FT_l))
#     self.lambdas_FT = lambdas_FT
#     self.dl_FT = dl_FT


                
# I = np.einsum('k,kql,q->l', phat.conj(), self.T_FT_l, phat)

# a = np.einsum('l,l', I**2, self.lambdas_FT**2) * self.dl_FT
# c = np.einsum('l,l', I**2, self.lambdas_FT**1) * self.dl_FT
# d = np.einsum('l,l', I**2, self.lambdas_FT**0) * self.dl_FT  

# T_vec_l = 2.0 * np.einsum('kql,q,l->kl', self.T_FT_l, phat, I)

# G = (1.0 / d * T_vec_l.dot(self.lambdas_FT**2)
#     - 2.0 * c / d**2 * T_vec_l.dot(self.lambdas_FT**1)
#     + (2.0 * c**2 / d**3 - 1.0 * a / d**2) * np.sum(T_vec_l, axis=1)) * self.dl_FT



# I = np.einsum('k,kql,q->l', phat.conj(), self.T_FT_l, phat)

# a = np.einsum('l,l', I**2, self.lambdas_FT**2) * self.dl_FT
# c = np.einsum('l,l', I**2, self.lambdas_FT**1) * self.dl_FT
# d = np.einsum('l,l', I**2, self.lambdas_FT**0) * self.dl_FT        

# L = a/d - (c/d)**2


                
# def G_long_FT(self, phat):
#     '''
#     Apply non-linear gradient function for longitudinal spread
#     to vector phat.
#     '''

#     I = np.einsum('k,kql,q->l', phat.conj(), self.T_FT_l, phat)

#     a = np.einsum('l,l', I**2, self.lambdas_FT**2) * self.dl_FT
#     c = np.einsum('l,l', I**2, self.lambdas_FT**1) * self.dl_FT
#     d = np.einsum('l,l', I**2, self.lambdas_FT**0) * self.dl_FT  

#     T_vec_l = 2.0 * np.einsum('kql,q,l->kl', self.T_FT_l, phat, I)

#     # G = 2.0 / d * (1.0 * np.einsum('kl,l->k', T_vec_l, self.lambdas_FT**2)
#     #                - 2.0 * c * np.einsum('kl,l->k', T_vec_l, self.lambdas_FT**1)
#     #                + (2.0 * c**2 / d - a / d) * np.einsum('kl->k', T_vec_l)) * self.dl_FT

#     G = (1.0 / d * T_vec_l.dot(self.lambdas_FT**2)
#         - 2.0 * c / d**2 * T_vec_l.dot(self.lambdas_FT**1)
#         + (2.0 * c**2 / d**3 - 1.0 * a / d**2) * np.sum(T_vec_l, axis=1)) * self.dl_FT

#     return G

# def L_long_FT(self, phat):
#     '''
#     Functional to be minimized giving longitudinal spread.
#     '''

#     I = np.einsum('k,kql,q->l', phat.conj(), self.T_FT_l, phat)

#     a = np.einsum('l,l', I**2, self.lambdas_FT**2) * self.dl_FT
#     c = np.einsum('l,l', I**2, self.lambdas_FT**1) * self.dl_FT
#     d = np.einsum('l,l', I**2, self.lambdas_FT**0) * self.dl_FT        

#     L = a/d - (c/d)**2

#     return L
