#!/usr/bin/env python

import numpy as np
from cmath import *

import pickle

from Utils import utils as ut
from Scattering import scattering2D as scat


def calc_t(S, dims, refmodes):
    '''
    Cut full t-matrices and considered parts of t-matrices from S-matrices.
    ''' 
    t = []
    t_q = []
    for i, s in enumerate(S):
        m = dims[i]/2
        n = min(m, refmodes)
        t.append( s[m:,0:m] )      # full t-matrix
        t_q.append( s[m:m+n,0:n] ) # considered part of t-matrix 

    return t, np.array(t_q)

def calc_tp(S, dims, refmodes):
    '''
    Cut full tp-matrices and considered parts of tp-matrices from S-matrices.
    ''' 
    tp = []
    tp_q = []
    for i, s in enumerate(S):
        m = dims[i]/2
        n = min(m, refmodes)
        tp.append( s[0:m,m:] )      # full tp-matrix
        tp_q.append( s[0:n,m:m+n] ) # considered part of tp-matrix 

    return tp, np.array(tp_q)

def calc_r(S, dims, refmodes):
    '''
    Cut full r-matrices and considered parts of r-matrices from S-matrices.
    ''' 
    r = []
    r_q = []
    for i, s in enumerate(S):
        m = dims[i]/2
        n = min(m, refmodes)
        r.append( s[0:m,0:m] )      # full r-matrix
        r_q.append( s[0:n,0:n] ) # considered part of r-matrix 

    return r, np.array(r_q)

def calc_rp(S, dims, refmodes):
    '''
    Cut full rp-matrices and considered parts of rp-matrices from S-matrices.
    ''' 
    rp = []
    rp_q = []
    for i, s in enumerate(S):
        m = dims[i]/2
        n = min(m, refmodes)
        rp.append( s[m:,m:] )      # full rp-matrix
        rp_q.append( s[m:m+n,m:m+n] ) # considered part of rp-matrix 

    return rp, np.array(rp_q)

def write_teigvals(t_q, refpos):
    '''
    Write t-eigenvalues at reference frequency to file.
    '''
    teigval = ut.sort_eig(t_q[3*(refpos)+1])[0]
    np.savetxt(data_direc+"teigvals."+filen+".dat", np.array([range(np.shape(teigval)[0]), teigval.real, teigval.imag]).transpose())

def print_freq_indep(qeigval, kmin, kmax, Dk):
    taumin = np.min( qeigval.real )
    taumax = np.max( qeigval.real )
    Dk = (kmax-kmin)/(Nk-1)
    print 
    print 'estimate for smallest frequency-independence'
    print 'wmin =', kmin, 'wmax =', kmax, 'dw =', Dk
    print 'tau_min =', taumin
    print 'tau_max =', taumax
    print 'frequency range: ', 1./(taumax-taumin)
    print 'in units of dw: ', 1./(taumax-taumin)/Dk
    print

def calc_eigensystem(a, modes):
    '''
    Calculcate eigenvalues, right eigenvectors and left eigenvectors of matrix a.
    Eigenvectors are store column-wise and inflated by appending zeros
    to size modes x modes.
    '''
    aeigval, aeigvec = ut.sort_eig(a)
    aeigvec  = aeigvec.transpose()
    aLeigvec = np.linalg.inv(aeigvec).transpose()
    aeigvec  = scat.inflate_small_mat(aeigvec, modes)
    aLeigvec = scat.inflate_small_mat(aLeigvec, modes)

    return aeigval, aeigvec, aLeigvec

def calc_qeigbas(q, Nk, modes, refmodes):
    '''
    Calculate right and left eigenvectors of q-operator at each frequency.
    '''
    #qeigbas = np.empty((Nk, refmodes, modes), dtype='complex')
    #qLeigbas = np.empty((Nk, refmodes, modes), dtype='complex')
    qeigbas = np.empty((Nk, modes, modes), dtype='complex')
    qLeigbas = np.empty((Nk, modes, modes), dtype='complex')
    for i in range(Nk):
        qeigbas[i] = (calc_eigensystem(q[i], modes)[1])#[0:refmodes]
        qLeigbas[i] = (calc_eigensystem(q[i], modes)[2])#[0:refmodes]

    return qeigbas, qLeigbas

def calc_qexpvals(eigvec, q, qLeigbas, modes, refmodes, Nk):
    '''
    Calculate proper expectation value of q using its left eigenbasis
    and standard deviations for vectors 'eigvec' at each frequency.
    '''
    times = np.empty((refmodes,Nk), dtype="complex")
    tisdv = np.empty((refmodes,Nk), dtype="complex")

    for w in range(Nk):
        for state in range(refmodes):
            vec = scat.inflate_vec(eigvec[state], modes)
            vecL = 1./np.linalg.norm(np.dot(qLeigbas[w], vec))**2 *\
                np.dot(qLeigbas[w].transpose(), np.dot(qLeigbas[w].conj(), vec.conj()))

            A   = scat.inflate_small_mat(q[w], modes)

            times[state][w] = np.dot(vecL, np.dot(A, vec)) 
            tisqr = np.dot(vecL, np.dot(A, np.dot(A, vec)))
            tisdv[state][w] = sqrt(tisqr - times[state][w]**2)

    return times, tisdv

def calc_transmitted_states(t, tinv, vec, Lvec, modes, refmodes, refpos, Nk):
    '''
    Calculate transmitted right vector at each frequency and transmitted
    left vector only at reference frequency. Transmitted right vector is
    normalized, transmitted left vector is normalized such that inner
    product of the two transmitted vectors is equal to one if vectors
    are aligned w.r.t. the considered modes and scattering into modes
    that are not considered is zero.
    '''
    vectransmit = np.zeros((Nk,refmodes,modes), dtype="complex")
    for i in range(Nk):
        t_inf = scat.inflate_small_mat(t[3*i+1], modes)
        for j in range(refmodes):
            vectransmit[i][j] = np.dot(t_inf, vec[j]) /\
                np.linalg.norm(np.dot(t_inf, vec[j]))

    vecLtransmit = np.zeros((refmodes,modes), dtype="complex")
    tfull_ref_inf = scat.inflate_small_mat(t[3*refpos], modes)
    tinv_inf = scat.inflate_small_mat(tinv, modes)
    for j in range(refmodes): 
        vecLtransmit[j] = np.dot(Lvec[j], tinv_inf) *\
            np.linalg.norm(np.dot(tfull_ref_inf, vec[j]))

    return vectransmit, vecLtransmit

def unpickle_states(filestr, modes, refmodes):
    '''
    Loads states from pickled files, calculates left states
    and inflates both.
    '''
    try:
        (vec, Lvec) = pickle.load(open(filestr, 'rb'))
        n,m = np.shape(vec)

        eigvec = np.identity(refmodes, dtype="complex")
        Leigvec = np.identity(refmodes, dtype="complex")
        eigvec[:n,:m] = vec
        Leigvec[:n,:m] = Lvec
        eigvec = scat.inflate_small_mat(eigvec, modes)
        Leigvec = scat.inflate_small_mat(Leigvec, modes)
    except:
        print "WARNING: unable to load pickled states"
        eigvec = np.identity(refmodes, dtype="complex")
        Leigvec = np.identity(refmodes, dtype="complex")
        eigvec = scat.inflate_small_mat(eigvec, modes)
        Leigvec = scat.inflate_small_mat(Leigvec, modes)
    
    return eigvec, Leigvec

def calc_mu(vectransmit, vecLtransmit, refmodes, Nk):
    '''
    Calculates measure mu for transverse change of
    output vector with frequency.
    '''
    mu = np.zeros((refmodes,Nk), dtype="float")
    for i in range(refmodes):
        for w in range(Nk):
            mu[i,w] = np.absolute(1-np.absolute(np.dot(vecLtransmit[i], vectransmit[w][i])))
                                  
    return mu
        
        
