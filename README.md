# Provable Low Rank Phase Retrieval

Provable Low Rank Phase Retrieval (AltMinLowRaP) implementation for solving a matrix of complex valued signals. This implementation is based on the paper "Provable Low Rank Phase Retrieval".

For more information: https://arxiv.org/abs/1902.04972


## Programs
The following is a list of which algorithms correspond to which Python script:

* custom_cgls_lrpr.py - Customized conjugate gradient least squares (CGLS) solver
* generate_lrpr.py - Generates sample measurements for testing
* image_tensor_small.npz - Sample image
* provable_lrpr.py - Implementation of provable LRPR
* reshaped_wirtinger_flow.py - Implementation of RWF
* provable_lrpr_run.py - Example on using provable LRPR implementation

## Tutorial
This tutorial can be found in provable_lrpr_run.py:

```
import numpy as np
import matplotlib.pyplot as plt
from provable_lrpr import provable_lrpr_fit
from generate_lrpr import generateLRPRMeasurements

# generating measurements
image_name = 'image_tensor_small.npz'
m_dim = 600
    
true_X, Y, A = generateLRPRMeasurements(image_name=image_name, m_dim=m_dim)

# parameters for LRPR
max_iters = 15

U_hat, B_hat = provable_lrpr_fit(Y=Y, A=A, max_iters=max_iters)

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
```

## Solution Example

<p align="center">
  <a href="url"><img src="https://github.com/soominkwon/Low-Rank-Phase-Retrieval/blob/main/provable_lrpr_example.png" align="left" height="300" width="300" ></a>
</p>

