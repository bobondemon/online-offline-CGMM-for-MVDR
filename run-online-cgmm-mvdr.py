'''
Psuedo-codes of using online CGMM with MVDR
'''
import numpy as np
from cgmm import PriorCGMM
import matplotlib.pyplot as plt
# import ...

# ===== Read file, do stft
...
stft_mat = ...
stft_mat_for_init = ...
# M: channel number
# K: CGMM cluster number (usually is 2)
# T: frame number
M, valid_n_fft, T = stft_mat.shape

# ===== Online CGMM MVDR
# Use stft_mat_for_init to initialize PriorCGMM
cgmmEngine = [PriorCGMM(stft_mat_for_init[:,i,:]) for i in range(valid_n_fft)]

# For each chunk, do MAP estimation to simulate online update
chunk_num = int(T/chunk_size)
for c in range(chunk_num):
  # ==== Online CGMM
  offset = chunk_size*c
  for i in range(valid_n_fft):
    cgmmEngine[i].run(stft_mat[:,i,offset:offset+chunk_size])
  # Get the spatial covariance matrix
  R = np.array([cgmmEngine[i].getR() for i in range(valid_n_fft)]) # (valid_n_fft, K, M, M)
  # Get the posterior results
  postArray = np.array([cgmmEngine[i].getPost() for i in range(valid_n_fft)]) # (valid_n_fft, K, T)
  if(c==1):
    mask_results = postArray
  else:
    mask_results = np.concatenate([mask_results,postArray],axis=2)
  # === MVDR
  Rv, Rx = R[:,0,:,:], R[:,1,:,:]
  # Do MVDR by using Rv and Rx
  stft_out_online = ... # (valid_n_fft, T)
  if(c==1):
    stft_out = stft_out_online
  else:
    stft_out = np.concatenate([stft_out,stft_out_online],axis=1)

# OLA back to wav form
wav_out=...

# ========== Plotting Area
plt.imshow(mask_results[:,1,:]) # plot the cluster index 1, as it represents speech cluster
plt.title('speech mask')
plt.show()
