#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 13:02:09 2021

@author: soominkwon

This program is for solving Low Rank Phase Retrieval by Vaswani et al.
(2016) via Conjugate Gradient Least Squares (CGLS).

The objective function is || C\sqrt{y} - A'Ub ||^2,
and we are minimizing for U.

Note that if CGLS solves equations of the form Ax = d, then in our case,

d = ( kron( b', A ) * C \sqrt{y} ). 

"""

import numpy as np

def construct_Y(A, U, B):
    """
        This function constructs y_hat = A * (Ub). Normally,
        this would be of dimensions (m x 1), but we construct this for
        each data sample, and thus is of dimensions (mq x 1).
        
        Arguments:
            A: Sampling vectors of dimensions (n x m x q).
            U: Basis matrix of dimensions (n x r)
            B: Matrix of dimensions (q x r)
    """
    
    n = A.shape[0]
    m = A.shape[1]
    q = A.shape[2]
    
    r = B.shape[1] # rank
    
    U = np.reshape(U, (n, r), order='F')
    
    Y_all = np.zeros((m*q, ))
    
    st = 0
    en = m
    
    for k in range(q):
        b_k = B[k]
        y_k = A[:, :, k].T @ U @ b_k
        Y_all[st:en] = y_k
    
        st+=m
        en+=m
        
    return Y_all


def construct_D(C_y, A, B):
    """
           
        Arguments:
            C_y: C * \sqrt{y} for all q (m*q x 1)
            A: Sampling vectors of dimensions (n x m x q).
            B: Matrix of dimensions (q x r)
    """    
    n = A.shape[0]
    m = A.shape[1]
    q = A.shape[2]
    
    r = B.shape[1] # rank
    
    solved_U = np.zeros((n*r, ))
    
    st = 0
    en = m
    for k in range(q):
        #A_y = A[:, :, k] @ C_y[st:en]
        #b_k_row = np.reshape(B[k], (1, -1))
        #B_kron = np.kron(b_k_row, A_y)
        #solved_U += B_kron.T.squeeze()
        b_k_row = np.reshape(B[k], (1, -1))
        b_kron_A = np.kron(b_k_row, A[:, :, k].T)
        b_kron_A = b_kron_A.T
        final = b_kron_A @ C_y[st:en]
        
        solved_U += final
        
        
        st += m
        en += m
        
    return solved_U



def cglsLRPR(A_sample, B_factor, C_y, max_iter=30, tol=1e-16):
    """
        Solves for U from the LRPR problem with modified CGLS for A(Ub) = y.
    """
    
    # initializing
    r = C_y
    s = construct_D(C_y=C_y, A=A_sample, B=B_factor)
    n = s.shape[0]
    x = np.zeros((n, )) # optimize variable
    

    # initializing for optimization
    p = s
    norms0 = np.linalg.norm(s)
    gamma = norms0**2
    normx = np.linalg.norm(x)**2
    xmax = normx
    
    iters = 0
    flag = 0
    
    while (iters < max_iter) and (flag == 0):
        #print('Current Iteration:', iters)
        
        q = construct_Y(A=A_sample, U=p, B=B_factor)
        
        delta = np.linalg.norm(q)**2
        
        alpha = gamma / delta
        
        # make updates
        x = x + alpha*p
        r = r - alpha*q
        
        # update s
        s = construct_D(C_y=r, A=A_sample, B=B_factor)
        
        norms = np.linalg.norm(s)
        gamma1 = gamma
        gamma = norms**2
        beta = gamma / gamma1
        
        # update p
        p = s + beta*p
        
        # convergence
        normx = np.linalg.norm(x)
        xmax = max(xmax, normx)
        flag = (norms <= norms0 * tol) or (normx * tol >= 1)
        
        iters += 1
    
    return x
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    