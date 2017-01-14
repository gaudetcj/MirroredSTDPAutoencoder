# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 20:07:14 2017

@author: Chase
"""

import cv2
import numpy as np
from keras.datasets import mnist


class MNIST():
    def __init__(self, resize=True):
        (X, _), (_, _) = mnist.load_data()
        self.X = X / 255.0
        self.resize = resize
                
    def _mean_shift(self, X):
        X = X - np.mean(X)
        return X
        
    def _on_off(self, X):
        on = np.maximum(0 , X)
        on = on.flatten()
        
        off = np.maximum(0, -X)
        off = off.flatten()
        
        on_off = np.concatenate((on, off))
        return on_off
        
    def get_poisson(self, tmax):
        sample = np.random.randint(0, self.X.shape[0])
        X = self.X[sample]
        
        if self.resize:
            X = cv2.resize(X, (14, 14))
            
        X = self._mean_shift(X)
        
        on_off = self._on_off(X)
        
        poisson = np.zeros((on_off.shape[0], tmax))
        for i in range(on_off.shape[0]):
            rands = np.random.random([tmax])
            for t in range(tmax):
                if (on_off[i] >= rands[t]):
                    poisson[i,t] = 1
        return poisson