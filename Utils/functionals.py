#!/usr/bin/env python

import numpy as np


class Functional:
    def __init__(self,
                 phi,
                 t,
                 phit_full,
                 wlist,
                 Dw,
                 opt_trans,
                 proj_list=None):

        w0 = 0.5 * (wlist[-2] + wlist[1])
        dw = w0 * 10**(-8)

        self.phi = phi

        self.phit = phit_full[1::3]
        self.dphit = (phit_full[2::3] - phit_full[0::3]) / (2*dw)
        self.wlist = wlist
        self.Dw = Dw

        self.wN = np.shape(wlist)[0]
        self.modes = np.shape(self.phit)[1]

        Pi = np.pi
        I = 0. + 1.J

        phit = self.phit
        dphit = self.dphit

        # matrices for longidtudinal spread
        self.T2 = np.einsum('ijk,ijl', dphit.conj(), dphit) * Dw
        self.T1 = (-I) * np.einsum('ijk,ijl', phit.conj(), dphit) * Dw
        self.T1 =  0.5 * (self.T1 + self.T1.conj().T) # gives smaller spreads
        self.T0 = np.einsum('ijk,ijl', phit.conj(), phit) * Dw

        # matrices for spectral correlation
        if (opt_trans==1):
            print "Calculating spectral autocorrelation functions"
            wN = np.shape(phit)[0]
            r_in = np.zeros((wN,), dtype='complex')
            r_out = np.zeros(np.shape(phit), dtype='complex')
            r_in[0] = np.einsum('w,w', phi.conj(), phi) * Dw
            r_out[0] = np.einsum('wlm,wln->mn', phit.conj(), phit) * Dw
            for o in (1+np.arange(wN-1)):
                r_in[o] = np.einsum('w,w', phi[:-o].conj(), phi[o:]) * Dw
                r_out[o] = np.einsum('wlm,wln->mn', phit[:-o].conj(), phit[o:]) * Dw
            self.r_in = r_in
            self.r_out = r_out

        # matrix for projector into specific modes
        if (proj_list.any()):
            print "Creating projector onto given subspace"
            Proj = np.diag(proj_list)
            self.P0 = np.einsum('wnm,nl,wlk->mk', phit.conj(), Proj, phit) * Dw


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

        a = np.dot( phat.conj(), np.dot( self.T2, phat))
        c = np.dot( phat.conj(), np.dot( self.T1, phat))
        d = np.dot( phat.conj(), np.dot( self.T0, phat))
        
        G = (-a/d**2 + 2.0*c**2/d**3) * np.dot(self.T0, phat) +\
            1.0/d * np.dot(self.T2, phat) -\
            2.0*c/d**2 * np.dot(self.T1, phat)

        return G 


    def L_long(self, phat):
        '''
        Functional to be minimized giving longitudinal spread.
        '''

        a = np.dot( phat.conj(), np.dot( self.T2, phat))
        c = np.dot( phat.conj(), np.dot( self.T1, phat))
        d = np.dot( phat.conj(), np.dot( self.T0, phat))

        L = a/d - (c/d)**2

        return L


    def G_corr_w(self, phat):
        '''
        Apply non-linear gradient function for optimal wave
        packets using squared difference of input and outpt
        spectral correlations.
        '''

        olist = np.arange(np.shape(self.r_out)[0]) * self.Dw

        rn_out = np.einsum('m,wmn,n->w', phat.conj(), self.r_out, phat)
        rc_out = np.einsum('m,wnm,n->w', phat.conj(), self.r_out.conj(), phat)
        d = np.dot( phat.conj(), np.dot( self.T0, phat))

        R_in = np.abs(self.r_in)**2/d**2
        R_out = np.abs(rn_out)**2/d**2

        dr_out = np.einsum('w,wkl,l->wk', rc_out, self.r_out, phat) +\
            np.einsum('w,wlk,l->wk', rn_out, self.r_out.conj(), phat)

        dR_out = 1.0/d**2 * dr_out -\
            2.0/d**3 * np.einsum('w,n->wn', np.abs(rn_out)**2, np.dot(self.T0, phat))

        G = 2.0 * np.einsum('w,wn->n', (R_out/R_out[0]-R_in/R_in[0]), dR_out) * self.Dw

        return G 


    def L_corr_w(self, phat):
        '''
        Functional to be minimized for optimal wave packets
        giving squared deviation of output correlation to
        input correlation.
        '''

        olist = np.arange(np.shape(self.r_out)[0]) * self.Dw

        rn_out = np.einsum('m,wmn,n->w', phat.conj(), self.r_out, phat)
        d = np.dot( phat.conj(), np.dot( self.T0, phat))

        R_in = np.abs(self.r_in)**2/d**2
        R_out = np.abs(rn_out)**2/d**2

        L = np.sum((R_out/R_out[0]-R_in/R_in[0])**2) * self.Dw

        return L


    def G_proj(self, phat):
        '''
        Apply non-linear gradient function for intensity
        scattered into specified subspace.
        '''

        p = np.dot( phat.conj(), np.dot( self.P0, phat))
        d = np.dot( phat.conj(), np.dot( self.T0, phat))
      

        #G = -1.0/d * np.dot(self.P0, phat) + \
        #    p/d**2 * np.dot(self.T0, phat)

        G = -1.0/(d-p) * np.dot(self.P0, phat) + \
            p/(d-p)**2 * (np.dot(self.T0, phat) - np.dot(self.P0, phat))

        return G 


    def L_proj(self, phat):
        '''
        Functional to be minimized giving for maximal intensity
        scattered into specified subspace.
        '''

        p = np.dot( phat.conj(), np.dot( self.P0, phat))
        d = np.dot( phat.conj(), np.dot( self.T0, phat))

        pi = np.dot( phat.conj(), np.dot( self.P0, phat))
        po = np.dot( phat.conj(), np.dot( np.eye(np.shape(self.P0)[0]) - self.P0, phat)) 

        #L = - p/d
        
        L = -p/(d-p)

        return L

 
