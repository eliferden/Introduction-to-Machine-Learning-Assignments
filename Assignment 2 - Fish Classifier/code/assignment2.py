# -*- coding: utf-8 -*-
"""
@author: Elif Erden

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

filename = input("Enter the name of the file..\n")
#filename = "data_training_set"
dataSet = pd.read_csv(filename + ".txt", header = None)
dataSet.columns = ['fishLength','fishClass']
dataSet['predictedClass'] = 0

salmon = dataSet[dataSet.fishClass == 1]
salmon = salmon.reset_index(drop=True)
seabass = dataSet[dataSet.fishClass == 2]
seabass = seabass.reset_index(drop=True)

#histogram for salmon
plt.hist(salmon.fishLength,color="red",label= "salmon")
plt.legend()
plt.xlabel("Fish Length(cm)")
plt.title("Histogram of Salmon Fish Lengths")
plt.show()
#histogram for seabass
plt.hist(seabass.fishLength,color="green",label= "seabass")
plt.legend()
plt.xlabel("Fish Length(cm)")
plt.title("Histogram of Seabass Fish Lengths")
plt.show()       
#boxplot for salmon 
plt.boxplot(salmon.fishLength)
plt.ylabel("Salmon Fish Length(cm)")
plt.show()
#boxplot for seabass
plt.boxplot(seabass.fishLength)
plt.ylabel("Seabass Fish Length(cm)")
plt.show()
#histogram for both in one plot
plt.hist(salmon.fishLength,color="red",label= "salmon", alpha=0.5)
plt.hist(seabass.fishLength,color="green",label= "seabass", alpha=0.5)
plt.legend()
plt.xlabel("Fish Length(cm)")
plt.title("Histogram of Salmon and Seabass Fish Lengths")
plt.legend()
plt.show() 

#converitng to arrays
dataSet = dataSet.values
salmon = salmon.values
seabass = seabass.values

#descriptive statistics 
salmonStd = np.std(salmon[:,0])
seabassStd = np.std(seabass[:,0])
salmonMean = np.mean(salmon[:,0])
seabassMean = np.mean(seabass[:,0])
print("Standard deviation for salmon: " + str(salmonStd))
print("Mean for salmon: " + str(salmonMean))
print("Standard deviation for seabass: " + str(seabassStd))
print("Mean for seabass: " + str(seabassMean))

#classifier function
def predict(dataSet):
    for i in range(len(dataSet)):
        probSalmon = scipy.stats.norm(307.45, 38.91).pdf(dataSet[i,0])
        probSeabass = scipy.stats.norm(345.79, 6.32).pdf(dataSet[i,0])
        if probSalmon > probSeabass:
           dataSet[i,2] = 1
        else:
           dataSet[i,2] = 2
      
    errorCounter = 0
    counter1 = 0
    counter2 = 0
    salmonCounter = 0
    seabassCounter = 0
    for j in range(len(dataSet)):
        if dataSet[j,1] != dataSet[j,2]:
             errorCounter += 1
        if dataSet[j,1] == 1 and dataSet[j,2] == 2:
             counter1 += 1
        if dataSet[j,1] == 2 and dataSet[j,2] == 1:
             counter2 += 1
        if dataSet[j,2] == 1:
             salmonCounter += 1
        if dataSet[j,2] == 2:
            seabassCounter += 1
    print("\nError count is " + str(errorCounter))
    print("Accuracy: " + str((len(dataSet)-errorCounter)/len(dataSet)*100))
    print("\nNumber of misclassifications of salmons: " + str(counter1))
    print("Number of misclassifications of seabasses: " + str(counter2))
    print("Number of correct of salmons: " + str(salmonCounter-counter2))
    print("Number of correct of seabasses: " + str(seabassCounter-counter1))

predict(dataSet)

