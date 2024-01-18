#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from statistics import mean

# read in CSV and convert to dataframe
datamk = 'magicKingdom.csv'
dfmk = pd.read_csv(datamk)

#print(dfmk.shape)
#dfmk.describe()

#print(dfmk)

# print prompt to user
print('Running machine learning algorithm, please wait for input prompt.\n')

# create ml model and variables to test and train with
modelmk = linear_model.LogisticRegression(max_iter = 10000)

y = dfmk.values[:, 1]
y = y.reshape(-1, 1)
X = dfmk.values[:, 0:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# fit ml model with training values
modelmk.fit(X_train, np.ravel(y_train))

# create log to test accuracy score with
y_pred_log = modelmk.predict(X_test)

#print(y_pred_log)

#print(metrics.accuracy_score(y_test, y_pred_log))

# prompt user for input, and then output ml prediction
def getUserInput():
    userInput = input("Enter a day of the week, ie...Mon, Tue, Wed, Thur, Fri, Sat, Sun. Q to quit.")
    
    if userInput == "Mon":
        predictionInput = [[0]]
        finalPrediction = modelmk.predict(predictionInput)
        finalPredictionStr = str(finalPrediction).strip("[]")
        print("The average wait time for rides on a Monday is " + finalPredictionStr + " minutes.\n")
        getUserInput()
        
    elif userInput == "Tue":
        predictionInput = [[1]]
        finalPrediction = modelmk.predict(predictionInput)
        finalPredictionStr = str(finalPrediction).strip("[]")
        print("The average wait time for rides on a Tuesday is " + finalPredictionStr + " minutes.\n")
        getUserInput()
        
    elif userInput == "Wed":
        predictionInput = [[2]]
        finalPrediction = modelmk.predict(predictionInput)
        finalPredictionStr = str(finalPrediction).strip("[]")
        print("The average wait time for rides on a Wednesday is " + finalPredictionStr + " minutes.\n")
        getUserInput()
        
    elif userInput == "Thur":
        predictionInput = [[3]]
        finalPrediction = modelmk.predict(predictionInput)
        finalPredictionStr = str(finalPrediction).strip("[]")
        print("The average wait time for rides on a Thursday is " + finalPredictionStr + " minutes.\n")
        getUserInput()
        
    elif userInput == "Fri":
        predictionInput = [[4]]
        finalPrediction = modelmk.predict(predictionInput)
        finalPredictionStr = str(finalPrediction).strip("[]")
        print("The average wait time for rides on a Friday is " + finalPredictionStr + " minutes.\n")
        getUserInput()
        
    elif userInput == "Sat":
        predictionInput = [[5]]
        finalPrediction = modelmk.predict(predictionInput)
        finalPredictionStr = str(finalPrediction).strip("[]")
        print("The average wait time for rides on a Saturday is " + finalPredictionStr + " minutes.\n")
        getUserInput()
        
    elif userInput == "Sun":
        predictionInput = [[6]]
        finalPrediction = modelmk.predict(predictionInput)
        finalPredictionStr = str(finalPrediction).strip("[]")
        print("The average wait time for rides on a Sunday is " + finalPredictionStr + " minutes.\n")
        getUserInput()
        
    elif userInput == "Q":
        return
    
    else:
        print("Not a valid input")
        getUserInput()
        
# call get user input method
getUserInput()


# In[94]:


bins = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

plt.hist(dfmk.waitTime, bins = bins)

plt.xticks(bins)

plt.ylabel('Number of Occurances')

plt.xlabel("Wait Times")

plt.title('Frequency of Posted Wait Times')

plt.show()


# In[97]:


waitTimeAvg = mean(dfmk.waitTime)
waitTimeAvgStr = str(waitTimeAvg)
underAverage = dfmk.loc[dfmk['waitTime'] < waitTimeAvg].count()[0]
aboveAverage = dfmk.loc[dfmk['waitTime'] > waitTimeAvg].count()[0]

labels = ['Under Average Wait Time', 'Above Average Wait Time']
colors = ['#a1a1a1', '#ff7654']

plt.pie([underAverage, aboveAverage], labels = labels, colors = colors, autopct='%.2f%%')

plt.title('Percentage of Posted Wait Times Under and Above Average')

plt.show()


# In[90]:


bins = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
monday = dfmk.loc[dfmk['day'] == 0]['waitTime']
tuesday = dfmk.loc[dfmk['day'] == 1]['waitTime']
wednesday = dfmk.loc[dfmk['day'] == 2]['waitTime']
thursday = dfmk.loc[dfmk['day'] == 3]['waitTime']
friday = dfmk.loc[dfmk['day'] == 4]['waitTime']
saturday = dfmk.loc[dfmk['day'] == 5]['waitTime']
sunday = dfmk.loc[dfmk['day'] == 6]['waitTime']

plt.figure(figsize=(10, 10))

labels = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

plt.boxplot([monday, tuesday, wednesday, thursday, friday, saturday, sunday], labels=labels)

plt.ylabel('Wait Time')
plt.title('Wait Time Comparisons by Day')
plt.show()


# In[ ]:




