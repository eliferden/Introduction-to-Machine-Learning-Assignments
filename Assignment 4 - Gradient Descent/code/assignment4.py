# -*- coding: utf-8 -*-
"""
@author: Elif Erden
"""
import math
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(xValues):
    p = []
    for i in xValues:
        p.append(1/(1 + math.exp(-i)))
    return p

xValues = np.arange(-10, 10, 0.1)
probabilities = sigmoid(xValues)
plt.plot(xValues,probabilities)
plt.title("Sigmoid function")
plt.show()
    