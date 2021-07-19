#!/usr/bin/env python
# coding: utf-8

# In[111]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[112]:


# Read data set data.csv
data=pd.read_csv('data.csv')
data.head()


# ## Variable Description
# Diagnosis: (M = malignant, B = benign)<br/>
# Ten real-valued features are computed for each cell nucleus<br/>
# radius (mean of distances from center to points on the perimeter)<br/>
# texture (standard deviation of gray-scale values)<br/>
# 
# perimeter<br/>
# area<br/>
# 
# smoothness (local variation in radius lengths)<br/>
# compactness (perimeter^2 / area - 1.0)<br/>
# concavity (severity of concave portions of the contour)<br/>
# concave points (number of concave portions of the contour)<br/>
# 
# symmetry<br/>
# 
# fractal dimension ("coastline approximation" - 1)<br/>

# In[113]:


# remove ('drop') unnecessary columns from the data set
data.drop(['Unnamed: 32','id'], inplace = True, axis = 1)


# In[114]:


# Check for missing values
data.info()
data.isna().sum()
#no missing values


# In[115]:


# replace 'M' with 0 and 'B' with 1 in the diagnosis column
data["diagnosis"] = [1 if i.strip() == "M" else 0 for i in data["diagnosis"]] 


# In[116]:


# Extract from data feature vector X and output label y
y = data.values[:, 0]
X = data.values[:, 1:]


# In[117]:


# Split data into X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)


# In[118]:


print('X_train.shape=', X_train.shape)
print('X_test.shape=', X_test.shape)
print('x_train.shape[0]+x_test.shape[0]=', X_train.shape[0] + X_test.shape[0])


# In[119]:


# Apply knn and evaluate clasifier by displaying the score; find best classifier
from sklearn.neighbors import KNeighborsRegressor

test_scores = []
train_scores = []
K = []

for k in range(1, 50):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    K.append(k)
    test_scores.append(knn.score(X_test, y_test))
    train_scores.append(knn.score(X_train, y_train))
    print('K=', K[k-1], ' test_score=', test_scores[k-1], '  train_test=', train_scores[k-1])

m = max(test_scores)
i = test_scores.index(m)
print('max test score: ', m, '  train score: ', train_scores[i], '   for K=', K[i])


# # Decision tree

# In[57]:


# Apply decision tree classifier and evaluate clasifier 
#by displaying the score, find best classifier


# In[120]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

clf = DecisionTreeClassifier(criterion = 'gini')
#chose gini over entropy
clf = clf.fit(X_train, y_train)

clf.get_depth()

y_pred = clf.predict(X_train)

from sklearn.metrics import accuracy_score
print("Accuracy of Decision Tree (X_train): ", accuracy_score(y_train, y_pred)*100, "%")

y_pred = clf.predict(X_test)
print("Accuracy of Decision Tree (X_test): ", accuracy_score(y_test, y_pred)*100, "%")


# In[121]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion = 'gini', max_depth=5, min_samples_leaf=3,
                            max_features='auto', random_state=2)
#chose gini over entropy

clg = clf.fit(X_train, y_train)

clf.get_depth()

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy of Decision Tree (test): ", accuracy_score(y_test, y_pred)*100, "%")

y_pred = clf.predict(X_train)
print("Accuracy of Decision Tree (train): ", accuracy_score(y_train, y_pred)*100, "%")


# # Random Forest

# In[138]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

forest_clf = RandomForestClassifier(max_depth=10, random_state=0) #past 10, the % stays same

forest_clf.fit(X_train, y_train)

print("Accuracy of Random Forest: ", forest_clf.score(X_test, y_test)*100, "%") 


# # Bagging

# In[152]:


from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

m = KNeighborsClassifier(n_neighbors=10)#change this
b = BaggingClassifier(m, n_estimators=80) #change up these numbers

b.fit(X_train, y_train)

print("Score = ", b.score(X_test, y_test)*100, "%") #%%


# In[ ]:




