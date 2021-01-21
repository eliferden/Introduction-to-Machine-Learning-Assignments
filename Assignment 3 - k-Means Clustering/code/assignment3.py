# -*- coding: utf-8 -*-
"""
@author: Elif Erden

"""
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import sys 

data = pd.read_csv("data3.txt", header = None)
dataSet = data.loc[:, 0:1]
dataSet[2] = 0
dataSet = dataSet.values
#plotting the original data
plt.scatter(dataSet[:,0],dataSet[:,1],color="orange",alpha=0.4)
plt.title("Dataset")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
#calculating min and max values of the dataset
maxX1 = max(dataSet[:,0]) 
maxX2 = max(dataSet[:,1]) 
minX1 = min(dataSet[:,0]) 
minX2 = min(dataSet[:,1])
print("min x1: " + str(minX1) + ", max x1: " + str(maxX1))
print("min x2: " + str(minX2) + ", max x2: " + str(maxX2))

k = int(input("Enter the number of clusters: \n"))
#generating random centroids by the number of clusters
clusterCenters = np.empty([k,3],dtype = float)
for i in range(k):
    randomX1 = random.uniform(minX1,maxX1)
    randomX2 = random.uniform(minX2,maxX2)
    #print("\nInitial centroid " + str(i+1) + "\nx1: " + str(randomX1) + ", x2: " + str(randomX2) + "\n")
    clusterCenters[i,0] = randomX1
    clusterCenters[i,1] = randomX2
    
iterationCounter = 0 
objectiveValue = 0
objective = []
stoppingCondition = False
counter = 0
while stoppingCondition == False:  
    #assigning observations to each cluster
    for j in range(len(dataSet)):  
       for i in range(k):
           distance = math.sqrt(pow(dataSet[j,0]-clusterCenters[i,0],2)+pow(dataSet[j,1]-clusterCenters[i,1],2))
           clusterCenters[i,2] = distance
       
       assignedCluster = 1 + np.argmin(clusterCenters[:,2])
       dataSet[j,2] = assignedCluster
       objectiveValue = objectiveValue + pow(min(clusterCenters[:,2]),2)
     
    objective.append(objectiveValue)
    colorSet = ["magenta","cyan","lime","orange","coral","blue","red","green","yellow","pink","aquamarine","olive","brown"]
    #checking if all clusters have at least one point or not
    for i in range(k):
       cluster = dataSet[dataSet[:,2] == i+1]
       if cluster.size == 0:
           counter += 1
    if counter == 0:       
        for i in range(k):          
            cluster = dataSet[dataSet[:,2] == i+1]
            plt.scatter(cluster[:,0],cluster[:,1],color=colorSet[i],alpha=0.4)
            plt.scatter(clusterCenters[i,0],clusterCenters[i,1],color="black")    
    else:
        sys.exit("[WARNING]: NO POINTS CAN BE ASSIGNED TO AT LEAST ONE CLUSTER!\nPLEASE TRY AGAIN..\n")
          
    #plotting the data with the cluster centers
    plt.title("Dataset with Clusters (Iteration = " + str(iterationCounter+1) + ")")
    plt.xlabel("x1")
    plt.ylabel("x2")
    #plt.savefig('plot' + str(iterationCounter+1) + '.png')
    plt.show()  
    
    previousClusterCenters = np.copy(clusterCenters)
    totalMovement = 0
    #updating the cluster centers
    for i in range(k):  
       meanX1 = np.average(dataSet[dataSet[:,2]==i+1,0])
       meanX2 = np.average(dataSet[dataSet[:,2]==i+1,1])
       clusterCenters[i,0] = meanX1
       clusterCenters[i,1] = meanX2
       totalMovement =  totalMovement + math.sqrt(pow(previousClusterCenters[i,0]-clusterCenters[i,0],2)+pow(previousClusterCenters[i,1]-clusterCenters[i,1],2))    
    
    objectiveValue = 0
    counter = 0
    print("Total movement at iteration " + str(iterationCounter+1) + " is " + str(totalMovement))
    iterationCounter += 1
    if totalMovement < 0.0001:       
        stoppingCondition = True
        
plt.plot(objective)
plt.title("Objective Function")
plt.xlabel("Iteration number")
plt.ylabel("Objective Function Value")
plt.show()
print("\nInitial objective function value: " + str(objective[0]) + "\nFinal objective function value: " + str(objective[iterationCounter-1]))

