#!/usr/bin/env python
# coding: utf-8

# In[13]:


from ucimlrepo import fetch_ucirepo
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import random


# In[14]:


seed = 43
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


# In[15]:


iris = fetch_ucirepo(id=53) 


# In[16]:


X = iris.data.features 
y = iris.data.targets


# In[17]:


y = y['class'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}).astype(np.float64)


# In[18]:


print ('The shape of X is:', X.shape)
print ('The shape of y is:', y.shape)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)


# In[20]:


model = Sequential(
    [
        Dense(32,activation = "relu"),
        Dense(8,activation = "relu"),
        Dense(3),
    ]
)


# In[21]:


model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
)


# In[22]:


model.fit(X_train,y_train,epochs=100, verbose=0)


# In[23]:


y_pred = model.predict(X_test)
for i in range(X_test.shape[0]):
    print( f"{y_test.values[i].astype(int)}, category: {np.argmax(y_pred[i])}")


# In[24]:


detected = np.sum(y_test.values.astype(int) == np.argmax(y_pred, axis=1))
accuracy = detected/X_test.shape[0]
print("Accuracy:",accuracy)


# In[ ]:




