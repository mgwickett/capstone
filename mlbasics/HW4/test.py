#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:22:15 2020

@author: madi
"""

import numpy as np 
from NN import *
from random_forest import * 

# dataset = load_breast_cancer() # load the dataset
np.random.seed(0)
x = np.random.randint(1, 50, (30, 426))
y = np.random.randint(2, 50, (1, 426))
yh = np.random.randint(2, 50, (1, 426))

nn = dlnet(x, y)
nn.nInit()
nn.forward(x)
nn.backward(y, yh)
nn.gradient_descent(x, y)

# n = 12330
# d = 17
# X = np.random.randint(1, 50, (n, d))
# y = np.random.randint(1, 50, (n, 1))
# rf = RandomForest()
# rf.fit(X, y)