#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:29:13 2021

@author: soominkwon
"""

import numpy as np


def initLambda(y, A):
    """
        Function to compute the norm of X to scale the initial vector.
    
        Arguments:
            y: observations (m x 1)
            A: sampling vectors (n x m)
            
        Returns:
            lambda_init: Lambda value to scale initial vector
    """

    n_dim = A.shape[0]
    m_dim = A.shape[1]
    
    numerator = n_dim * m_dim
    denominator = 0
    
    # computing the variance
    for i in range(m_dim):
        a_i = A[:, i]
        l1_norm = np.linalg.norm(a_i, ord=1)
        denominator += l1_norm
        
    first_term = numerator / denominator
    
    # computing mean of y
    second_term = y.mean()
        
    lambda_init = first_term * second_term
    
    return lambda_init


def initSpectral(y, A, lambda_init, alpha_l, alpha_u):
    """
        Spectral initialization for RWF.
        
        Arguments:
            y: observations (m x 1)
            A: sampling vectors (n x m)
            lambda_init: Lambda value computed from initLambda() function
            alpha_l: Lower cutoff bound
            alpha_u: Upper cutoff bound
            
        Returns:
            z_init: Initial guess for RWF
    """
    
    # initializing dimnsions
    n_dim = A.shape[0]
    m_dim = A.shape[1]
    
    # computing upper and lower bounds
    upper_bound = alpha_u*lambda_init
    lower_bound = alpha_l*lambda_init

    y_lower = np.where(y<lower_bound, 0, y)
    y_trunc = np.where(y_lower>upper_bound, 0, y_lower)
    
    Y_mat = np.zeros((n_dim, n_dim), dtype=np.complex)
    
    for i in range(m_dim):
        a_i = A[:, i]
        a_i = np.reshape(a_i, (-1, 1))
        
        per_slice = y_trunc[i] * (a_i @ a_i.T)
        Y_mat += per_slice
        
    Y_mat = (1/m_dim)*Y_mat
    
    # computing eigenvectors
    eig_val, eig_vec = np.linalg.eig(Y_mat)
    
    # taking top eigenvector
    z_tilde = eig_vec[:, 0]
    
    # scaling eigenvector
    z_init = lambda_init * z_tilde
    
    return z_init


def computeGradient(z, y, A):
    """
        Computing gradient for the gradient loop.
        
        Arguments:
            z: Solved vector
            y: Observation vector
            A: Sampling matrix 
            
        Returns:
            updated_z: Updated z for t-th iteration.
    """
    
    m_dim = A.shape[1]
    update_z = np.zeros(z.shape, dtype=np.complex)

    for i in range(m_dim):
        a_i = A[:, i]
        a_i = np.reshape(a_i, (-1, 1))
        
        y_i = y[i]
        
        first_term = a_i.T @ z

        second_term = y_i * (first_term / np.abs(first_term))
        sum_term = (first_term - second_term) * a_i

        update_z += sum_term.squeeze()

    updated_z = (1/m_dim) * update_z
    
    return updated_z


    
def rwf_fit(y, A, max_iterations=50, mu=0.8, alpha_l=1, alpha_u=5, print_iter=False):
    """
        Training loop for RWF. Currently only supports a fixed number of iterations without a 
        tolerance value. Parameters are set to values reported by paper.
        
        Returns:
            z_init: Solved z after a total of T iterations.
    """
    
    # spectral initialization
    init_lambda = initLambda(y=y, A=A)
    z_init = initSpectral(y=y, A=A, lambda_init=init_lambda, alpha_l=alpha_l, alpha_u=alpha_u)    
    
    for t in range(max_iterations):
        
        if print_iter:
            print('Current Iteration:', t)
        
        current_grad = computeGradient(z=z_init, y=y, A=A)
        
        # updating z
        z_init -= mu*current_grad
        
    return z_init



        
        
        
        
        
        
        
        
        
        
        
        

