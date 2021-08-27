# Provable Low Rank Phase Retrieval

Provable Low Rank Phase Retrieval (AltMinLowRaP) implementation for solving a matrix of complex valued signals. This implementation is based on the paper "Provable Low Rank Phase Retrieval".

For more information: https://arxiv.org/abs/1902.04972


## Programs
The following is a list of which algorithms correspond to which Python script:

* custom_cgls_lrpr.py - Customized conjugate gradient least squares (CGLS) solver
* provable_lrpr.py - Implementation of provable LRPR
* reshaped_wirtinger_flow.py - Implementation of RWF
* sample_run.py - Example on using provable LRPR implementation

## Tutorial
This tutorial can be found in sample_run.py:

```
import numpy as np
import matplotlib.pyplot as plt
from provable_lrpr import provable_lrpr_fit

# importing sample data
data_name = 'mouse_small_data.npz'

with np.load(data_name) as sample_data:
    vec_X = sample_data['arr_0']
    Y = sample_data['arr_1']
    A = sample_data['arr_2']
    
# initializing parameters
image_dims = [10, 30]
rank = 1
iters = 5

# fitting new X
X_lrpr =  provable_lrpr_fit(Y=Y, A=A, max_iters=iters, rank=rank)

X = np.reshape(vec_X, (image_dims[0], image_dims[1], -1), order='F')
X_lrpr = np.reshape(X_lrpr, (image_dims[0], image_dims[1], -1), order='F')

# plotting results
plt.imshow(np.abs(X[:, :, 0]), cmap='gray')
plt.title('True Image')
plt.show()

plt.imshow(np.abs(X_lrpr[:, :, 0]), cmap='gray')
plt.title('Reconstructed Image via Provable LRPR')
plt.show()
```

## Solution Example

<p align="center">
  <a href="url"><img src="https://github.com/soominkwon/Provable-Low-Rank-Phase-Retrieval/blob/main/provable_sample_results.png" align="left" height="300" width="300" ></a>
</p>

