#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:03:27 2021

@author: soominkwon
"""


import numpy as np

def generateLRPRMeasurements(image_name, m_dim):
    """ Function to obtain measurements y's (m x q) and A's (m x n x q).
    
        Arguments:
            image_name: name of .npz file to load (n1 x n2 x q)
            m_dim: dimensions of m
    
    """
    
    with np.load(image_name) as data:
        tensor = data['arr_0']
        
    q_dim = tensor.shape[2]
    vec_images = np.reshape(tensor, (-1, q_dim))
    
    n_dim = vec_images.shape[0]
    
    A_tensor = np.random.randn(n_dim, m_dim, q_dim)
    Y = np.zeros((m_dim, q_dim))
    
    for k in range(q_dim):
        A_k = A_tensor[:, :, k]
        x_k = vec_images[:, k]
        
        norm_x_k = np.linalg.norm(x_k)
        x_k = x_k / norm_x_k
        
        y_k = A_k.T @ x_k
        Y[:, k] = y_k
        
    Y = np.abs(Y)
    
    return tensor, Y, A_tensor 


#image_name = 'image_tensor_small.npz'
#m_dim = 2000
    
#Y, A = generateLRPRMeasurements(image_name=image_name, m_dim=m_dim)
#np.savez('lrpr_variables.npz', Y, A)