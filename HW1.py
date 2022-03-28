### 
# In the "Wine Data Set", there are 3 types of wines and 13 different 
# features of each instance -> [178*14] matrix (178 instances and 14 
# features). In this HW, we will implement the MAP of the classifier
# for 54 instances with their features.
#
# Goal: determine which type of wine according to the given features
# 
# 1. Split dataset to train and test datasets, for each type of wine, 
#    randomly split 18 for testing. train_set:[124*14] test_set:[54*14]
# 2. Evaluate posterior probabilities with likelihood fcns and prior 
#    distribution of the training set.
# 3. Calculate the accuracy rate of the MAP detector (should exceed 90%)
# 4. Plot visualized result of the "testing data" (with built in PCA fcn)
###

import pandas as pd
import numpy as np

# independent features with gaussian distrobution (14 features)
features = ["Wine type","Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", 
            "Total phenols", "Flavanoids", "Non Flavanoid phenols", "Proanthocyanins",
            "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

# Read .csv file  
wineDataSet = pd.read_csv('Wine.csv', header=None)
# Add title to dataset
# wineDataSet.columns = features

# Find the number of different wines in the data set
sets_of_wine = [0,0,0]
for i in range(wineDataSet.shape[0]):
    sets_of_wine[int(wineDataSet[0].values[i])-1]+=1

# Wine 1: 1-59, Wine 2: 60-130, Wine 3: 131-178
wine1 = wineDataSet.iloc[               :sets_of_wine[0]                ,:]
wine2 = wineDataSet.iloc[sets_of_wine[0]:sets_of_wine[0]+sets_of_wine[1],:]
wine3 = wineDataSet.iloc[sets_of_wine[0]+sets_of_wine[1]:               ,:]

# Sample the wine according to its label 
wine1_test = wine1.sample(n=18)
wine1_train = wine1.drop(wine1_test.index)
wine2_test = wine2.sample(n=18)
wine2_train = wine2.drop(wine2_test.index)
wine3_test = wine3.sample(n=18)
wine3_train = wine3.drop(wine3_test.index)

# Build train and test sets 
train_data = pd.concat([wine1_train, wine2_train, wine3_train])
test_data = pd.concat([wine1_test, wine2_test, wine3_test])


# Find the number of different wines in the training set
sets_of_wine = [0,0,0]
for i in range(train_data.shape[0]):
    sets_of_wine[train_data[0].values[i]-1]+=1

# Calculate mean and standard deviation of each feature of different wine. 
# Stored as [mean_feature1, stdev_feature1], ... (13 for each wine)
feature_distribution = []
data_accum = 0
for i in range(3):
    for j in range(13):
        # calculate mean and stdev for train_data[0:sets_of_wine[i]][j+1]
        mean  = np.average(train_data[j+1].values[data_accum:data_accum+sets_of_wine[i]])
        stdev = np.std(train_data[j+1].values[data_accum:data_accum+sets_of_wine[i]])
        feature_distribution.append([mean,stdev])
    data_accum += sets_of_wine[i]

# Calculate prior for each wine
priors = [0,0,0]
train_total = sets_of_wine[0]+sets_of_wine[1]+sets_of_wine[2]
for i in range(3):
    priors[i] = sets_of_wine[i]/train_total

# Compute posteriors for MAP



# # print(train_data[1].values[0:15])
# print(train_data)
# print(train_data[1].values[0:10])

# # Access test
# test = (1,5)
# test2 = (2,10)
# l = [test,test2]
# print(l[0][1])