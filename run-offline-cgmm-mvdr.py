'''
Psuedo-codes of using offline CGMM with MVDR
'''
import numpy as np
import matplotlib.pyplot as plt
from cgmm import CGMM
# import ...

# ===== Read file, do stft
...
stft_mat = ...
# M: channel number
# K: CGMM cluster number (usually is 2)
# T: frame number
M, valid_n_fft, T = stft_mat.shape

# ===== Offline CGMM
cgmmEngine = [CGMM(stft_mat[:,i,:]) for i in range(valid_n_fft)]
for i in range(valid_n_fft):
  cgmmEngine[i].run()
# Get the spatial covariance matrix
R = np.array([cgmmEngine[i].getR() for i in range(valid_n_fft)]) # (valid_n_fft, K, M, M)
# Get the posterior results
mask_results = np.array([cgmmEngine[i].getPost() for i in range(valid_n_fft)])  # (valid_n_fft, K, T)

# ===== MVDR
Rv, Rx = R[:,0,:,:], R[:,1,:,:]
# Do MVDR by using Rv and Rx
stft_out = ... # (valid_n_fft, T)

# OLA back to wav form
wav_out=...

# ========== Plotting Area
# mask_results: (valid_n_fft, K=2, T)
plt.imshow(mask_results[:,1,:]) # plot the cluster index 1, as it represents speech cluster
plt.title('speech mask')
plt.show()