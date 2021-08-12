#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:35:50 2021

@author: soominkwon
"""

import functools
import numpy as np
from scipy.optimize import minimize
from generate_lrpr import generateLRPRMeasurements

def LRPRinit(rank, y, A):
    """ Function to use spectral initialization for the factor matrices as described in
        Vaswani et al. (2017).
    
        Arguments:
            rank: Rank of X
            y: Observation matrix with dimensions (m x q)
            A: Measurement tensor with dimensions (m x n x q)

    """    
    
    # initializing
    m_dim = y.shape[0]
    q_dim = y.shape[1]
    n_dim = A.shape[1]
    
    Y_u = np.zeros((n_dim, n_dim))
    trunc_val = 9 * Y.mean()
        
    # looping through each frame
    for k in range(q_dim):
        per_y = y[:, k]
        y_trunc = np.where(per_y<trunc_val, per_y, 0)
        Y_u += A[:, :, k].T @ np.diag(y_trunc) @ A[:, :, k]
    
    Y_u = (1/(m_dim*q_dim)) * Y_u
    U, S, V = np.linalg.svd(Y_u)
    U_init = U[:, :rank]
    
    B_init = np.zeros((q_dim, rank))
    
    # initializing B
    for k in range(q_dim):
        A_u = A[:, :, k] @ U_init
        Y_b = A_u.T @ np.diag(Y[:, k]) @ A_u
        U, S, V = np.linalg.svd(Y_b)
        B_init[k] = np.sqrt(Y[:, k].mean()) * U[:, 0]
    
    dictionary = {}
    dictionary[0] = U_init
    dictionary[1] = B_init

    return dictionary


def updateC(A, U, B):
    """ Function to update the diagonal phase matrix C.
    
        Arguments: 
            A: Measurement tensor with dimensions(m x n x q)
            U: Basis matrix with dimensions (n x r)
            B: Matrix with dimensions (q x r)
            
        Returns:
            C_tensor: Tensor where the frontal slices represent C_k (diagonal phase matrix)
                        with dimensions (m x m x q)
    """
    
    m_dim = A.shape[0]    
    q_dim = B.shape[0]
    
    C_tensor = np.zeros((m_dim, m_dim, q_dim))
    
    for k in range(q_dim):
        A_k = A[:, :, k]
        b_k = B[k]
        
        x_hat = U @ b_k
        y_hat = A_k @ x_hat
        
        phase_y = np.exp(1j*np.angle(y_hat))
        #phase_y = np.sign(y_hat)
        C_k = np.diag(phase_y)
        C_tensor[:, :, k] = C_k
        
        
    return C_tensor



def updateU(U, B, y, A, C):
    
    objective_func = 0
    
    q_dim = y.shape[1]
    
    # reshaping U
    U = np.reshape(U, (A.shape[1], B.shape[1]))
    
    for k in range(q_dim):
        per_y = y[:, k]
        per_A = A[:, :, k]
        b_k = B[k]
        C_k = C[:, :, k]
        
        sqrt_y = np.sqrt(per_y)
        
        first_term = C_k @ sqrt_y
        second_term = per_A @ (U @ b_k)
        
        slice_obj_func = np.linalg.norm(first_term - second_term)**2
        
        objective_func += slice_obj_func
    return objective_func



def updateEachB(b_k, U, y_k, A_k, C_k):
    
    sqrt_y = np.sqrt(y_k)
    
    first_term = C_k @ sqrt_y
    second_term = A_k @ (U @ b_k)
        
    obj_func = np.linalg.norm(first_term - second_term)**2
    
    return obj_func



def LRPRfit(rank, y, A, tol=1e-4, max_iterations=20):
    
    # initializing factor matrices
    factor_mats = LRPRinit(rank=rank, y=y, A=A)
    print('Spectral Initialization Complete.')

    U_init = factor_mats[0]
    B_init = factor_mats[1]
    q_dim = y.shape[1]

    iterations = 0
    error = 1
    current_error = 0

    
    while (error > tol) and (iterations < max_iterations):
        print('Current Iteration:', iterations)
        prev_error = current_error
        
        # update C
        C_tensor = updateC(A=A, U=U_init, B=B_init)

        # solve for U
        solve_U = functools.partial(updateU, B=B_init, y=y, A=A, C=C_tensor)
        U_solver = minimize(solve_U, U_init, method='CG', tol=1e-4)
        updated_U = np.reshape(U_solver.x, (U_init.shape))
        U_init = updated_U
        
        print('Update U Completed.')
        
        # update for each b_k
        for k in range(q_dim):
            y_k = y[:, k]
            A_k = A[:, :, k]
            C_k = C_tensor[:, :, k]
            b_k = B_init[k]
            
            solve_B = functools.partial(updateEachB, U=U_init, y_k=y_k, A_k=A_k, C_k=C_k)
            B_solver = minimize(solve_B, b_k, method='CG', tol=1e-4)
            updated_B = np.reshape(B_solver.x, (1, -1))
            B_init[k] = updated_B
            
            print('Update B', k, 'Completed.')
            
        # put updated factors back into dictionary
        factor_mats[0] = U_init
        factor_mats[1] = B_init

        
        # terminate for any one of the factor conditions
        current_error = U_solver.fun
        error = abs(prev_error - current_error)
        iterations += 1
        
        print('Objective Function Value:', U_solver.fun)
        print('Current Difference in Error Value: '+ str(error) + str('\n'))
        
        
    return factor_mats
        
        
    
image_name = 'image_tensor_small.npz'
m_dim = 2000
rank = 3

X, Y, A = generateLRPRMeasurements(image_name=image_name, m_dim=m_dim)

with np.load('lrpr_variables.npz') as data:
    Y = data['arr_0']
    A = data['arr_1']

factor_mats = LRPRfit(rank=rank, y=Y, A=A)

np.savez('lrpr_U', factor_mats[0])
np.savez('lrpr_B', factor_mats[1])


    
        
 
    




