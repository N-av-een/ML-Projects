#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv(r"D:\Tensor Code\Machine Learning\Machine Learning R_27.07.21\Machine Learning Project 2 - Bitcoin Price Prediction\bitcoin.csv",encoding="ISO-8859-1")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.isna().sum()


# In[6]:


df.describe()


# In[7]:


df.drop(["Date"],axis=1,inplace=True)


# In[8]:


df.head()


# In[9]:


PredictionDays=30
#Create another column shifted "n" units up
df["Predictions"]=df[["Price"]].shift(-PredictionDays)
df.head()


# In[10]:


df.tail()


# In[11]:


#Create independent data set
#Here we will convert the data frame into numpy array  and drop the prediction COlumn
x=np.array(df.drop(["Predictions"],axis=1))


# In[12]:


x.shape


# In[13]:


x


# In[14]:


#Remove the last 'n' rows where "n" is the PredcitionDays
x=x[:len(df)-PredictionDays]
x


# In[15]:


x.shape


# In[16]:


#Create the dependent data set
#Convert the dataframe into numpy array
y=np.array(df["Predictions"])
#Get all values except last "n" rows
y=y[:-PredictionDays]
y


# In[17]:


y.shape


# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[19]:


PredictionDays_array=np.array(df.drop(["Predictions"],axis=1))[-PredictionDays:]
print(PredictionDays_array)


# In[20]:


from sklearn.svm import SVR
svr_rbf=SVR(kernel="rbf",C=1e3,gamma=0.0001)
svr_rbf.fit(X_train,y_train)


# In[21]:


##Printing Predicted Values
svm_prediction=svr_rbf.predict(X_test)
print(svm_prediction)
print()
print(y_test)


# In[22]:


#Print the model preditions for the next 30 Days
print("model preditions for the next 30 Days")
svm_prediction=svr_rbf.predict(PredictionDays_array)
print(svm_prediction)
print()
#Print the actual price the bitcoin for the next 30 days
print(df.tail(PredictionDays))


# In[ ]:




