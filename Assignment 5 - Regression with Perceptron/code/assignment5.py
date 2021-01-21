# -*- coding: utf-8 -*-
"""
@author: Elif Erden

"""
# gradient descent
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
import math

data = pd.read_csv("data_regression1.txt", header = None)
#data = pd.read_csv("data_regression2.csv", header = None)
data[2] = 0.0
data = data.values
n = len(data)/2
trainSet = data[:int(n)]
testSet = data[int(n):]
#plotting the original data
plt.scatter(data[:,0],data[:,1],color="orange",alpha=0.4)
plt.title("Dataset")
plt.xlabel("Feature")
plt.ylabel("Output")
plt.show()

w0 = rd.uniform(-0.01,0.01)
w1 = rd.uniform(-0.01,0.01)

trainLoss = []
testLoss = []
eta = 0.005 #dataset1
#eta = 0.00005 #dataset2
prevLoss = 1000000000
trloss = 0
teloss = 0
stoppingCondition = False
while (stoppingCondition == False):
    for i in range(len(trainSet)):
        trainSet[i,2] = w0 + w1*trainSet[i,0]
        testSet[i,2] = w0 + w1*testSet[i,0]
        w0 = w0 + eta*(trainSet[i,1]-trainSet[i,2])
        w1 = w1 + eta*(trainSet[i,1]-trainSet[i,2])*trainSet[i,0]
    #calculating the loss function
    for i in range(len(trainSet)):
        trloss = trloss + (1/2)*math.pow(trainSet[i,1]-trainSet[i,2],2)
        teloss = teloss + (1/2)*math.pow(testSet[i,1]-testSet[i,2],2)
    trainLoss.append(trloss)  
    testLoss.append(teloss)
    if (prevLoss - trloss < 0.0001):
        stoppingCondition = True
    prevLoss = trloss
    trloss = 0
    teloss = 0
#plotting the loss functions
plt.plot(trainLoss,color="blue",label="Training Set")
plt.plot(testLoss,color="orange",label="Test Set")
plt.title("Loss function")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()
#plotting the original data with the regression line
plt.scatter(data[:,0],data[:,1],color="orange",alpha=0.4)
for i in range(len(data)):
    data[i,2] = w0+w1*data[i,0]
    
plt.plot(data[:,0],data[:,2],color="black")
plt.title("Dataset")
plt.xlabel("Feature")
plt.ylabel("Output")
plt.show()


