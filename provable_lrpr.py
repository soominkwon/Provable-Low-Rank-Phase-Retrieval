#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:44:28 2021

@author: soominkwon
"""

import numpy as np
from custom_cgls_lrpr import cglsLRPR
from reshaped_wirtinger_flow import rwf_fit


def chooseRank(array, omega=1.3):
    """
        Function to return the index of the difference between
        the j-th and (n)-th element with threshold omega.
    """
    
    array_len = array.shape[0]
    lambda_n = array[array_len-1]
    
    idx = 0
    
    for i in range(array_len):
        diff = np.abs(array[i] - lambda_n)
    
        if diff > omega:
            idx = i
            
    # if rank 1 was chosen
    if idx == 0:
        idx = 1
    return idx


def LRPRinit(Y, A, rank=None):
    """ Function to use spectral initialization for the basis matrix U, where
        X = UB.
    
        Arguments:
            Y: Observation matrix with dimensions (m x q)
            A: Measurement tensor with dimensions (n x m x q)
            rank: Rank for U. If rank is None, then choose a rank by
                    the method specified in the paper.

    """    
    
    # initializing
    m = Y.shape[0]
    q = Y.shape[1]
    n = A.shape[0]
        
    Y_u = np.zeros((n, n), dtype=np.complex)
    trunc_val = 9*Y.mean() # value for truncation
    
    # looping through each frame
    for k in range(q):
        y_k = Y[:, k]
        trunc_y_k = np.where(np.abs(y_k)<=trunc_val, y_k, 0)
        Y_u += A[:, :, k] @ np.diag(trunc_y_k) @ A[:, :, k].conj().T
        
    # normalizing factors
    Y_u = (1/(m*q)) * Y_u
    
    # computing SVD for Y
    eig_val, eig_vec = np.linalg.eig(Y_u)
    
    # choosing rank if not given
    if rank is None:
        rank = chooseRank(eig_val)
        
        # fixing the chosen rank if needed
        max_rank = min(n, q)
        if rank > max_rank:
            rank = max_rank
        
        U = eig_vec[:, :rank]
    else:
        U = eig_vec[:, :rank]
    
    print('Chosen Rank:', rank)
    print('Spectral Initialization Complete.')
    
    return U
    

def updateC(A, U, B):
    """ Function to update the diagonal phase matrix C.
    
        Arguments: 
            A: Measurement tensor with dimensions(n x m x q)
            U: Basis matrix with dimensions (n x r)
            B: Matrix with dimensions (q x r)
            
        Returns:
            C_tensor: Tensor where the frontal slices represent C_k (diagonal phase matrix)
                        with dimensions (m x m x q)
    """
    
    m_dim = A.shape[1]    
    q_dim = B.shape[0]
    
    C_tensor = np.zeros((m_dim, m_dim, q_dim), dtype=np.complex)
    
    for k in range(q_dim):
        A_k = A[:, :, k]
        b_k = B[k]
        
        x_hat = U @ b_k
        y_hat = A_k.conj().T @ x_hat
        
        phase_y = np.exp(1j*np.angle(y_hat))
        #phase_y = np.sign(y_hat)
        C_k = np.diag(phase_y)
        C_tensor[:, :, k] = C_k
               
    return C_tensor


def provable_lrpr_fit(Y, A, max_iters, rank=None, print_iter=True):
    """
        Training loop for provable LRPR.
        
        Arguments:
            Y: Observation matrix (m x q)
            A: Sampling tensor (n x m x q)
            max_iters: Maximum number of iterations for training loop
            
        Returns:
            U_init: Solved U after iteration T
            B_init: Solved B after iteration T
        
    """
    
    # initializing U and B    
    U_init = LRPRinit(Y=Y, A=A, rank=rank)
    
    # initializing dimensions
    m = A.shape[1]
    q = A.shape[2]    
    n, r = U_init.shape
    
    B_init = np.zeros((q, r), dtype=np.complex)

    Y = np.sqrt(Y) # square rooting Y
    
    # starting training loop
    for i in range(max_iters):
        
        if print_iter:
            print('Current Iteration:', i)
        
        # solving RWF for b_k
        for k in range(q):
            y_k = Y[:, k]
            A_k = A[:, :, k]
            
            A_hat = U_init.conj().T @ A_k
            b_k = rwf_fit(y=y_k, A=A_hat)
            B_init[k] = b_k
            
        # updating phase matrix C
        C_all = updateC(A, U_init, B_init)

        # applying QR decomposition
        Qb, Rb = np.linalg.qr(B_init)
        B_init = Qb
         
        # update U
        st = 0
        en = m
        Y_vec = np.zeros((m*q, ), dtype=np.complex)

        for k in range(q):
            C_y = C_all[:, :, k] @ Y[:, k]
            Y_vec[st:en] = C_y
        
            st += m
            en += m        
        
        U_vec = cglsLRPR(A_sample=A, B_factor=B_init, C_y=Y_vec)
        U_init = np.reshape(U_vec, (n, r), order='F')
        
        # applying QR decomposition
        Qu, Ru = np.linalg.qr(U_init)
        U_init = Qu
        

    X_lrpr = U_init @ B_init.conj().T
    return X_lrpr
        
  
