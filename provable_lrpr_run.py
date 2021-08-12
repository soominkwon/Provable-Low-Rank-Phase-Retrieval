#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:22:45 2021

@author: soominkwon
"""

import numpy as np
import matplotlib.pyplot as plt
from provable_lrpr import lrpr_fit
from generate_lrpr import generateLRPRMeasurements

# generating measurements
image_name = 'image_tensor_small.npz'
m_dim = 2500
    
true_X, Y, A = generateLRPRMeasurements(image_name=image_name, m_dim=m_dim)

# parameters for LRPR
rank = 5
max_iters = 20

U_hat, B_hat = lrpr_fit(rank=rank, Y=Y, A=A, max_iters=max_iters)

# reconstructing X and plotting
img_row = true_X.shape[0]
img_col = true_X.shape[1]
q = B_hat.shape[0]

solved_vec_X = U_hat @ B_hat.T

solved_X = np.reshape(solved_vec_X, (img_row, img_col, q))

# plotting first image
plt.imshow(np.abs(true_X[:, :, 0]), cmap='gray')
plt.title('True Image')
plt.show()

plt.imshow(np.abs(solved_X[:, :, 0]), cmap='gray')
plt.title('Solved Image via Provable LRPR')
plt.show()