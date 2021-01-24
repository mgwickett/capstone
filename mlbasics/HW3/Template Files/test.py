#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:56:45 2020

@author: madi
"""

from imgcompression import * 
from pca import *
from regression import *
from nb import *
import numpy as np 

# color image 
np.random.seed(0)
test = np.random.randint(1, 50, size = (100, 20, 3))
u, s, v = ImgCompression().svd(test)
ImgCompression().rebuild_svd(u, s, v, 10)
ImgCompression().compression_ratio(test, 10)
ImgCompression().recovered_variance_proportion(s, 10)

# bw image 
# bw = np.ones((4, 5))
# u, s, v = ImgCompression().svd(bw)
# ImgCompression().rebuild_svd(u, s, v, 3)


# PCA testing
# data = np.array([[11, 2, 7, 5], [8, 4, 6, 12], [10, 2, 9, 3]])
# pca = PCA()
# pca.fit(data)
# pca.transform_rv(data)

# Regression testing
# x = np.array([0, 1, 2])
# reg = Regression()
# reg.construct_polynomial_feats(x, 4)
# reg.ridge_fit_closed(data, x, 0.757)

# # i need a bigger dataset
# xtrain = np.random.randint(1, size = (100, 7))
# ytrain = np.random.randint(1, size = 100)
# reg.ridge_cross_validation(xtrain, ytrain)


# # naive bays
# nb = NaiveBayes()
# xh = nb.XH
# xm = nb.XM
# nb._likelihood_ratio(xh, xm)