#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

'''
This is the implementation with CGMM based on paper "Online MVDR Beamformer Based on Complex Gaussian Mixture Model with Spatial Prior for Noise Robust ASR"
We use Circularly Symmetric Gaussian Mixture Model (Complex domain with mean=0 and pseudocovariance=0)
# M: channel number
# K: CGMM cluster number (usually is 2)
# T: frame number
'''

class CGMM:
  def __init__(self,Y,K=2,openAssert=False):
    self._openAssert = openAssert
    self._K = K # number of clusters (number of sound sources + 1 background noise)
    self._Y = Y # Y: (M = mic_number or feat_dim, T = frame_num)
    self._M, self._T = Y.shape
    M, T = self._M, self._T
    # declares the parameters shape and type
    self._Phi = np.zeros([K,T],dtype=complex) # (K,T): variance of signals w.r.t. all time frames for each clusters
    self._R = np.zeros([K,M,M],dtype=complex) # (K,M,M): covariances for each clusters
    self._invR = np.zeros([K,M,M],dtype=complex) # (K,M,M): inverse covariances for each clusters
    self._alpha = np.zeros([K,]) # (K,): mixture weights
    self._posterior = np.zeros([K,T]) # posterior prob.
    self._steerVec = np.zeros([K,]) # steering vector

    self._initParam()

  def _initParam(self):
    K = self._K
    Y = self._Y # Y: (M, T)
    M, T = self._M, self._T

    if K==2:
      self._R[0,...] = 1e-6*np.eye(M).astype(complex) # indicates noise cluster
      self._R[1,...] = np.matmul(Y,Y.conj().T)/T # (M,M), indicates speech cluster
    else:
      # WARNING: Bad performance.
      rand_scale = 1e-3*np.random.rand(K).astype(complex) # (K,)
      rand_eye = np.einsum('k,ij->kij',rand_scale,np.eye(M).astype(complex))
      for k in range(1,K):
        self._R[k,...] = np.matmul(Y,Y.conj().T)/T
      self._R += rand_eye
    
    self._invR = np.linalg.inv(self._R)
    tmpMat = np.einsum('mt,kmn->knt',Y.conj(),self._invR)
    self._Phi = np.einsum('knt,nt->kt',tmpMat,Y)/M # (K,T)
    self._alpha = np.ones([K,])/K

  def getR(self):
    return np.copy(self._R)
  def getPost(self):
    return np.copy(self._posterior)
  def getPhi(self):
    return np.copy(self._Phi)
  def getMixWeights(self):
    return np.copy(self._alpha)

  def _calLogGaussianProb(self,Y,Phi,R,invR):
    """
    Arguments: (for nfft: num_fft, M: num_mics, T: num_frames)
        Y: (M, T), T observations with M-dim
        Phi: (T,), Representing the signal variance for each time t. Precise definition can be found in paper.
        R, invR: (M,M), the spatial (inverse) covariance matrix
    Return:
        logProb: (T,), the log-probabilities
    """
    M, T = Y.shape
    R = (R + np.transpose(np.conj(R))) / 2
    if self._openAssert:
      tmpMat = np.einsum('mt,mn->nt',Y.conj(),invR)
      tmpMat = np.einsum('nt,nt->t',tmpMat,Y)
      assert(np.allclose(np.real(tmpMat),np.real(Phi)*M))
      assert(np.allclose(np.imag(tmpMat),np.imag(Phi)*M))

    det = np.linalg.det(R).real
    logProb = -M*np.log(Phi*np.pi) - np.log(det) - M
    if self._openAssert:
      assert(np.allclose(np.imag(logProb),0))

    return logProb

  def run(self,itr_num=10):
    """
    Maximal Likelihood (ML) with EM algorithm
        itr_num: iteration number
    Return:
        post: (K,T), posterior probabilities for T observations (K-dim)
    """
    K, M, T, Y = self._K, self._M, self._T, self._Y
    R, invR, Phi, alpha, post = self._R, self._invR, self._Phi, self._alpha, self._posterior
    log_post = np.zeros(post.shape)

    for itr in range(itr_num):

      # ===== E Step
      # log_post, post: (K,T)
      log_alpha = np.log(alpha) # (K,)
      for k in range(K):
        log_post[k,:] = log_alpha[k] + self._calLogGaussianProb(Y,Phi[k,:],R[k,...],invR[k,...])
      post = np.exp(log_post)
      post = post/np.sum(post,axis=0)
      if self._openAssert:
        assert(np.allclose(np.sum(post,axis=0),1))
      post_sum = np.sum(post,axis=1) # (K,)

      # ===== M Step
      # Update Phi
      tmpMat = np.einsum('mt,kmn->knt',Y.conj(),invR)
      Phi = np.einsum('knt,nt->kt',tmpMat,Y)/M # (K,T)
      # Update R
      tmpMat = np.einsum('kt,mt->kmt',(post/Phi),Y) # (K,M,T)
      R = np.einsum('kmt,tn->kmn',tmpMat,Y.T.conj()) # (K,M,M)
      R = np.einsum('kmn,k->kmn',R,1/post_sum)
      invR = np.linalg.inv(R)
      # Update alpha. It is not updated in paper. Can comment below line.
      alpha = post_sum/T

    # Compute post after all iterations
    log_alpha = np.log(alpha) # (K,)
    for k in range(K):
      log_post[k,:] = log_alpha[k] + self._calLogGaussianProb(Y,Phi[k,:],R[k,...],invR[k,...])
    post = np.exp(log_post)
    post = post/np.sum(post,axis=0)
    self._R, self._invR, self._Phi, self._alpha, self._posterior = R, invR, Phi, alpha, post
    return post


'''
This is the implementation with spatial prior CGMM based on paper "Online MVDR Beamformer Based on Complex Gaussian Mixture Model with Spatial Prior for Noise Robust ASR"
# M: channel number
# K: CGMM cluster number (usually is 2)
# T: frame number
'''
class PriorCGMM(CGMM):
  def __init__(self,Y,K=2,openAssert=False):
    CGMM.__init__(self,Y,K,openAssert)
    CGMM.run(self,itr_num=3)
    # Init Super-parameters
    # See https://en.wikipedia.org/wiki/Conjugate_prior for conjugate prior
    self._Eta = self._T # Control the ratio of previous v.s. new data
    # We use lambda (see definition in paper) instead of usual super-parameters in inverse-wishart
    # self._posterior is of shape (K,T)
    self._Lambda = np.sum(self._posterior,axis=1) # (K,)
    assert(len(self._Lambda)==K)

  def run(self,Y,itr_num=3):
    """
    Maximal A Posterior (MAP) with EM algorithm
        itr_num: iteration number
    Return:
        post: (K,T), posterior probabilities for T observations (K-dim)
    """
    self._Y = Y # Y: (M, T), set the new data as current Y
    M, T = Y.shape
    assert(M==self._M)
    self._T = T # set the new frame number as current T
    K = self._K
    R, invR, Phi, alpha, post = self._R, self._invR, self._Phi, self._alpha, self._posterior
    log_post = np.zeros(post.shape)

    for itr in range(itr_num):

      # ===== E Step
      # log_post, post: (K,T)
      log_alpha = np.log(alpha) # (K,)
      for k in range(K):
        log_post[k,:] = log_alpha[k] + self._calLogGaussianProb(Y,Phi[k,:],R[k,...],invR[k,...])
      post = np.exp(log_post)
      post = post/np.sum(post,axis=0)
      if self._openAssert:
        assert(np.allclose(np.sum(post,axis=0),1))
      post_sum = np.sum(post,axis=1) # (K,)

      # ===== M Step
      # Update Phi
      tmpMat = np.einsum('mt,kmn->knt',Y.conj(),invR)
      Phi = np.einsum('knt,nt->kt',tmpMat,Y)/M # (K,T)
      # # Update alpha
      # alpha = post_sum/T
      # Update R, MAP udpate
      lambda_next = self._Lambda + post_sum # (K,)
      tmpConst = (self._Eta + M + 1)/2
      numerator = self._Lambda + tmpConst # (K,)
      demonimator = lambda_next + tmpConst # (K,)
      tmpMat = np.einsum('kt,mt->kmt',(post/Phi),Y) # (K,M,T)
      tmpMat = np.einsum('kmt,tn->kmn',tmpMat,Y.T.conj()) # (K,M,M)
      # R: (K,M,M)
      priorInfo = np.einsum('k,kmn->kmn',numerator/demonimator,R) # (K,M,M)
      newInfo = np.einsum('k,kmn->kmn',1/demonimator,tmpMat) # (K,M,M)
      R = priorInfo + newInfo
      invR = np.linalg.inv(R)

    # Compute post after all iterations
    log_alpha = np.log(alpha) # (K,)
    for k in range(K):
      log_post[k,:] = log_alpha[k] + self._calLogGaussianProb(Y,Phi[k,:],R[k,...],invR[k,...])
    post = np.exp(log_post)
    post = post/np.sum(post,axis=0)
    post_sum = np.sum(post,axis=1) # (K,)
    self._R, self._invR, self._Phi, self._alpha, self._posterior = R, invR, Phi, alpha, post

    # update super-parameters
    self._Eta = self._Eta + T
    self._Lambda = self._Lambda + post_sum # (K,)

    return post

