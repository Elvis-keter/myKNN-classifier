#!/usr/bin/env python
# coding: utf-8

# In[135]:


from sklearn import datasets


# In[136]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance


# In[137]:


def e(a,b):
    return distance.euclidean(a,b)


# In[138]:


class uns_knn():
    def fit(self, features_train, labels_train):
        self.features_train = features_train
        self.labels_train = labels_train
        
    def predict(self, features_test):
        predictclosest = []
        for item in features_test:
            label = self.closest(item)
            predictclosest.append(label)
            
        return predictclosest
    
    def closest(self, item):
        nearest = e(item, self.features_train[0])
        nearest_index = 0
        for i in range(1, len(self.features_train)):
            distance = e(item, self.features_train[i])
            if distance < nearest:
                nearest = distance
                nearest_index = i
                
        return self.labels_train[nearest_index]


# In[139]:


iris = datasets.load_iris()


# In[140]:


features = iris.data


# In[141]:


labels = iris.target


# In[142]:


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.5)


# In[143]:


elvis_classifier = uns_knn()
#elvis_classifier = KNeighborsClassifier()


# In[144]:


elvis_classifier.fit(features_train, labels_train)


# Verginica product

# In[147]:


prediction = elvis_classifier.predict(features_test)


# In[148]:


print(accuracy_score(labels_test, prediction))


# In[149]:


virgin = [[7.1, 2.9, 5.5, 2.4]]
virgin_prediction = elvis_classifier.predict(virgin)


# In[150]:


if virgin_prediction[0] == 0:
    print("Setosa")
    


# In[151]:


if virgin_prediction[0] == 1:
    print("Versicolour")


# In[152]:


if virgin_prediction[0] == 2:
    print("Virginica")

