# -*- coding: utf-8 -*-
"""
@author: Elif Erden

"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    
 dataSet = pd.read_csv("iris_data.txt", header = None)
 #train set
 trainSet = dataSet.loc[np.r_[0:30, 50:80, 100:130], [0, 3, 4]]
 trainSet = trainSet.reset_index(drop=True)
 trainSet.columns = ['sepalLength','petalWidth','irisName']
 #test set
 testSet = dataSet.loc[np.r_[30:50, 80:100, 130:150], [0, 3, 4]]
 testSet = testSet.reset_index(drop=True)
 testSet.columns = ['sepalLength','petalWidth','irisName']
 #plotting the train set
 setosa = trainSet[trainSet.irisName == "Iris-setosa"]
 versicolor = trainSet[trainSet.irisName == "Iris-versicolor"]
 virginica = trainSet[trainSet.irisName == "Iris-virginica"]

 plt.scatter(setosa.sepalLength,setosa.petalWidth,color="red",label= "setosa")
 plt.scatter(versicolor.sepalLength,versicolor.petalWidth,color="green",label= "versicolor")
 plt.scatter(virginica.sepalLength,virginica.petalWidth,color="blue",label= "virginica")
 plt.legend()
 plt.title("Sepal Length versus Petal Width")
 plt.xlabel("Sepal Length(cm)")
 plt.ylabel("Petal Width(cm)")
 plt.show()
 
 #setting datasets by default values
 trainSet ['distance'] = 0
 testSet ['predictedLabel'] = ''
 
 #getting distance metric from the user
 distanceMetric = input("Choose the distance metric to be used\nEnter 0 for Euclidean distance\nEnter 1 for Manhattan distance\nEnter 2 for Cosine Distance\n>> ")
 kNNclassify(trainSet,testSet,distanceMetric)
 
def kNNclassify(trainSet,testSet,distanceMetric):
    
 if distanceMetric == '0' or distanceMetric == '1' or distanceMetric == '2':
  
  kValue = int(input("Enter the k value for the number of neighbors: "))
  plotDecisionBoundaries(kValue) 
  
  for i in range(len(testSet)):
        
     for j in range(len(trainSet)):
    
       if distanceMetric == '0':
                
          distance = math.sqrt(pow(trainSet.iloc[j,0]-testSet.iloc[i,0],2)+pow(trainSet.iloc[j,1]-testSet.iloc[i,1],2))
          trainSet.iloc[j,trainSet.columns.get_loc('distance')] = distance
        
       elif distanceMetric == '1':
          
          distance = abs(trainSet.iloc[j,0]-testSet.iloc[i,0])+abs(trainSet.iloc[j,1]-testSet.iloc[i,1])
          trainSet.iloc[j,trainSet.columns.get_loc('distance')] = distance
        
       elif distanceMetric == '2':
          
          distance = 1 - ((trainSet.iloc[j,0]*testSet.iloc[i,0]+trainSet.iloc[j,1]*testSet.iloc[i,1])/(math.sqrt(pow(testSet.iloc[i,0],2)+pow(testSet.iloc[i,1],2))*math.sqrt((pow(trainSet.iloc[j,0],2)+pow(trainSet.iloc[j,1],2)))))
          trainSet.iloc[j,trainSet.columns.get_loc('distance')] = distance
     
     #sorting distances in ascending order
     trainSet = trainSet.sort_values(['distance'], ascending=[1])
     #identifying the labels
     setosaCounter = 0
     versicolorCounter = 0
     virginicaCounter = 0
    
     for c in range(kValue):
        
        if trainSet.iloc[c,2] == 'Iris-setosa':
            setosaCounter += 1
            
        if trainSet.iloc[c,2] == 'Iris-versicolor':
            versicolorCounter += 1
            
        if trainSet.iloc[c,2] == 'Iris-virginica':
            virginicaCounter += 1
    
     var = {setosaCounter: "Iris-setosa",versicolorCounter:"Iris-versicolor",virginicaCounter:"Iris-virginica"}
     predictedLabel = var.get(max(var))
     testSet.iloc[i,testSet.columns.get_loc('predictedLabel')] = predictedLabel
     
  print("")     
  print(testSet)
  getAccuracy(testSet)    
  
 else: 
  print("Wrong input entered!")
  
def getAccuracy(testSet):
    
    errorCounter = 0
    for e in range(len(testSet)):
        
        if testSet.iloc[e,2] != testSet.iloc[e,3]:
            
            errorCounter += 1
    
    print("\nError count: " + str(errorCounter) + "/" + str(len(testSet)))
    print("Accuracy(%): " + str((len(testSet)-errorCounter)/len(testSet)*100))

def plotDecisionBoundaries(kValue):

   from matplotlib.colors import ListedColormap
   from sklearn import neighbors
   
   df = pd.read_csv("iris_data.txt", header = None)
   X = df.loc[np.r_[0:30, 50:80, 100:130], [0, 3]]
   X = X.values

   y = df.loc[np.r_[0:30, 50:80, 100:130], [4]]
   y.columns = ['irisName']
   for i in range(len(y)):
    
     if y.iloc[i,y.columns.get_loc('irisName')] == 'Iris-setosa':
         y.iloc[i,y.columns.get_loc('irisName')] = 0
      
     if y.iloc[i,y.columns.get_loc('irisName')] == 'Iris-versicolor':
         y.iloc[i,y.columns.get_loc('irisName')] = 1
      
     if y.iloc[i,y.columns.get_loc('irisName')] == 'Iris-virginica':
         y.iloc[i,y.columns.get_loc('irisName')] = 2
       
   y['irisName'] = y['irisName'].astype(int)  
   y = y.values.ravel()

   h = .02

   cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
   cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

   for metric in ['euclidean', 'manhattan']:

       clf = neighbors.KNeighborsClassifier(kValue,metric=metric)
       clf.fit(X, y)
       
       x_min = X[:, 0].min() - 1
       x_max = X[:, 0].max() + 1
       y_min = X[:, 1].min() - 1
       y_max = X[:, 1].max() + 1
       xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
       Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

       Z = Z.reshape(xx.shape)
       plt.figure()
       plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
       
       plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
       plt.xlim(xx.min(), xx.max())
       plt.ylim(yy.min(), yy.max())
       plt.title("Decision boundaries (k = " + str(kValue) + ", distance metric = " + metric + ")")
       plt.xlabel("Sepal Length(cm)")
       plt.ylabel("Petal Width(cm)")
       plt.show()
        
main()