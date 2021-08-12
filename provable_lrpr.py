#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:44:28 2021

@author: soominkwon
"""


import numpy as np
from custom_cgls_lrpr import cglsLRPR
from reshaped_wirtinger_flow import rwf_fit


def LRPRinit(rank, Y, A):
    """ Function to use spectral initialization for the basis matrix U, where
        X = UB.
    
        Arguments:
            rank: Rank of X
            Y: Observation matrix with dimensions (m x q)
            A: Measurement tensor with dimensions (n x m x q)

    """    
    
    # initializing
    m = Y.shape[0]
    q = Y.shape[1]
    n = A.shape[0]
    
    # squaring Y
    Y = Y**2
    
    Y_u = np.zeros((n, n))
    trunc_val = 9*Y.mean() # value for truncation
    
    # looping through each frame
    for k in range(q):
        y_k = Y[:, k]
        trunc_y_k = np.where(np.abs(y_k)<=trunc_val, y_k, 0)
        Y_u += A[:, :, k] @ np.diag(trunc_y_k) @ A[:, :, k].T
        
    # normalizing factors
    Y_u = (1/(m*q)) * Y_u
    
    # computing SVD for Y
    U, S, Vh = np.linalg.svd(Y_u, full_matrices=True)
    U_init = U[:, :rank]

    return U_init
    

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
    
    C_tensor = np.zeros((m_dim, m_dim, q_dim))
    
    for k in range(q_dim):
        A_k = A[:, :, k]
        b_k = B[k]
        
        x_hat = U @ b_k
        y_hat = A_k.T @ x_hat
        
        #phase_y = np.exp(1j*np.angle(y_hat))
        phase_y = np.sign(y_hat)
        C_k = np.diag(phase_y)
        C_tensor[:, :, k] = C_k
               
    return C_tensor


def lrpr_fit(rank, Y, A, max_iters, print_iter=True):
    """
        Training loop for provable LRPR.
        
        Arguments:
            rank: Rank for matrix X
            Y: Observation matrix (m x q)
            A: Sampling tensor (n x m x q)
            
        Returns:
            U_init: Solved U after iteration T
            B_init: Solved B after iteration T
        
    """
    
    # initializing dimensions
    n = A.shape[0]
    m = A.shape[1]
    q = A.shape[2]    
    
    # initializing U and B    
    U_init = LRPRinit(rank=rank, Y=Y, A=A)
    B_init = np.zeros((q, rank))
    
    print('Spectral Initialization Complete.')
    
    
    # starting training loop
    for i in range(max_iters):
        
        if print_iter:
            print('Current Iteration:', i)
        
        # solving RWF for b_k
        for k in range(q):
            y_k = Y[:, k]
            A_k = A[:, :, k]
            
            A_hat = U_init.T @ A_k
            b_k = rwf_fit(y=y_k, A=A_hat)
            B_init[k] = b_k
            
        # updating phase matrix C
        C_all = updateC(A, U_init, B_init)
        
        # applying QR decomposition
        Qb, Re = np.linalg.qr(B_init)
        B_init = Re
         
        # update U
        st = 0
        en = m
        Y_vec = np.zeros((m*q, ))

        for k in range(q):
            C_y = C_all[:, :, k] @ Y[:, k]
            Y_vec[st:en] = C_y
        
            st += m
            en += m        
        
        U_vec = cglsLRPR(A_sample=A, B_factor=B_init, C_y=Y_vec)
        U_init = np.reshape(U_vec, (n, rank), order='F')
        
        # applying QR decomposition
        Qu, Ru = np.linalg.qr(U_init)
        U_init = Qu
        
    return U_init, B_init
        
            





