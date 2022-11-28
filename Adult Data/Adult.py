#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"D:\Tensor Code\Machine Learning\Machine Learning R_27.07.21\Machine Learning Project 1 - Adult Salary Prediction\adult_data.csv",encoding="ISO-8859-1")


# In[3]:


df.head()


# In[4]:


df.columns=["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","salary"]


# In[10]:


df.head()


# In[11]:


for i in df.columns:
    print("Column Names",i,"has total unique values of",len(df[i].unique()))
    print(df[i].unique())
    print("_"*100)


# In[12]:


df["capital_gain"].unique()


# In[13]:


df.dtypes


# In[ ]:





# In[14]:


def handle_capital_gain(df):
    df["capital_gain"]=np.where(df["capital_gain"]==0,np.nan,df["capital_gain"])
    df["capital_gain"]=np.log(df["capital_gain"])
    df["capital_gain"]=df["capital_gain"].replace(np.nan,0)


# In[15]:


handle_capital_gain(df)


# In[16]:


df["capital_gain"].dtypes


# In[17]:


plt.figure(dpi=100)
sns.histplot(df["capital_gain"],kde=True)
plt.grid(visible=True,linestyle="-.",linewidth=0.5,color="grey",alpha=0.5)


# In[18]:


df.describe()


# In[19]:


df.isnull().sum()


# In[20]:


df["salary"].unique()


# Removing Outlier from hours_per_week

# In[21]:


print("Hours per Week","\n")
print(df["hours_per_week"].unique(),"\n")
print("Total of unique values::",df["hours_per_week"].nunique())


# In[22]:


plt.figure(dpi=100)
sns.histplot(df["hours_per_week"],kde=True)
plt.grid(visible=True,linewidth=0.5,linestyle="-",color="grey",alpha=0.5)


# In[23]:


plt.figure(dpi=100)
sns.boxplot(df["hours_per_week"])


# In[24]:


df.hours_per_week.describe()


# In[25]:


def remove_outlier_hours_per_week(df):
    IQR=df["hours_per_week"].quantile(0.75)-df["hours_per_week"].quantile(0.25)
    
    Q1=df["hours_per_week"].quantile(0.25)-(1.5*IQR)
    Q3=df["hours_per_week"].quantile(0.75)+(1.5*IQR)
    
    df.loc[df["hours_per_week"]<=Q1,"hours_per_week"]=Q1
    df.loc[df["hours_per_week"]>=Q3,"hours_per_week"]=Q3


# In[26]:


remove_outlier_hours_per_week(df)


# In[27]:


plt.figure(dpi=80)
sns.boxplot(df["hours_per_week"])


# <u>**Education_num**<u>

# In[28]:


plt.figure(dpi=80)
sns.histplot(df.education_num,kde=True)


# In[31]:


sns.boxplot(df.education_num)


# In[32]:


def remove_outlier_education_num(df):
    IQR=df["education_num"].quantile(0.75)-df["education_num"].quantile(0.25)
    
    Q1=df["education_num"].quantile(0.25)-(1.5*IQR)
    Q3=df["education_num"].quantile(0.75)+(1.5*IQR)
    
    df.loc[df["education_num"]<=Q1,"education_num"]=Q1
    df.loc[df["education_num"]>=Q3,"education_num"]=Q3


# In[33]:


remove_outlier_education_num(df)


# In[34]:


plt.figure(dpi=80)
sns.set_style("darkgrid")
sns.boxplot(df["education_num"],color="grey")


# <u>**Capital_loss**<u>

# In[35]:


plt.figure(dpi=80)
sns.histplot(df["capital_loss"],kde=True)


# In[36]:


sns.boxplot(df["capital_loss"])


# In[37]:


def remove_outliers_capital_loss(df):
    IQR=df["capital_loss"].quantile(0.75)-df["capital_loss"].quantile(0.25)
    
    Q1=df["capital_loss"].quantile(0.25)-(1.5*IQR)
    Q3=df["capital_loss"].quantile(0.75)+(1.5*IQR)
    
    df.loc[df["capital_loss"]<=Q1,"capital_loss"]=Q1
    df.loc[df["capital_loss"]>=Q3,"capital_loss"]=Q3


# In[38]:


remove_outliers_capital_loss(df)


# In[39]:


plt.figure(dpi=80)
sns.boxplot(df["capital_loss"])


# In[40]:


for i in df.columns:
    print("Column Names",i,"has total unique values of",len(df[i].unique()))
    print(df[i].unique())
    print("_"*100)


# In[41]:


df.dtypes


# In[42]:


def feature_engineering(df):
    
    #df["salary"]=df["salary"].replace(">50K",">50K")
    #df["salary"]=np.where(df["salary"]>"50K",1,0) # <=50K
    
    df['salary'] = df['salary'].replace(' >50K', '>50K')
    df['salary'] = np.where(df['salary'] > '50K', 1, 0)
    
    df["sex"]=np.where(df["sex"]=="Male",1,0)
    
    label_enco_race={value:key for key,value in enumerate(df["race"].unique())}
    df["race"]=df["race"].map(label_enco_race)
    
    label_enco_relation={value:key for key,value in enumerate(df["relationship"].unique())}
    df["relationship"]=df["relationship"].map(label_enco_relation)
    
    df["occupation"]=np.where(df["occupation"]=="?","Missing",df["occupation"])
    label_enco_occupation={value: key for key,value in enumerate(df["occupation"].unique())}
    df["occupation"]=df["occupation"].map(label_enco_occupation)
    
    label_enco_marital_status={value:key for key,value in enumerate(df["marital_status"].unique())}
    df["marital_status"]=df["marital_status"].map(label_enco_marital_status)  
    
    label_enco_edu={value:key for key,value in enumerate(df["education"].unique())}
    df["education"]=df["education"].map(label_enco_edu)
    
    df["workclass"]=np.where(df["workclass"]=="?","Missing",df["workclass"])
    label_enco_workclass={value:key for key,value in enumerate(df["workclass"].unique())}
    df["workclass"]=df["workclass"].map(label_enco_workclass)
    
    df["native_country"]=np.where(df["native_country"]=="?","Missing",df["native_country"])
    label_enco_native_country={value:key for key,value in enumerate(df["native_country"].unique())}
    df["native_country"]=df["native_country"].map(label_enco_native_country)


# In[43]:


plt.figure(dpi=100)
corr=df.corr()
sns.heatmap(corr,annot=True)


# In[44]:


plt.figure(dpi=100)
mask=np.triu(np.ones_like(corr))
sns.heatmap(corr,mask=mask,annot=True)


# In[45]:


df=df.drop("fnlwgt",axis=1)


# In[46]:


df.columns


# In[47]:


df.isnull().sum()


# In[48]:


df.head()


# # After running feature engineering

# In[49]:


feature_engineering(df)


# In[50]:


df.head()


# In[51]:


df["salary"].value_counts()


# # <u>Standardisation<u>

# In[52]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[53]:


sc=StandardScaler()


# In[54]:


X = df[['age', 'workclass', 'education', 'education_num', 'marital_status', 'occupation'
, 'relationship','race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']]


# In[55]:


y=df["salary"]


# In[56]:


y.value_counts()


# In[58]:


X=sc.fit_transform(X)


# In[59]:


X


# In[60]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# In[61]:


print("Train data shape:{}".format(X_train.shape))
print("Test data shape:",X_test.shape)


# In[62]:


from sklearn.linear_model import LogisticRegression


# In[63]:


lg=LogisticRegression()


# In[64]:


lg.fit(X_train,y_train)


# In[65]:


y_pred=lg.predict(X_test)


# In[66]:


result={
    "Actual":y_test,
    "Predicted":y_pred
}


# In[67]:


pd.DataFrame(result)


# In[68]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[70]:


print("Accuracy_score","\n",accuracy_score(y_test,y_pred))
print("\n","Confusion_matrix","\n",confusion_matrix(y_test,y_pred))
print("\n","Classification Report:\n","\n",classification_report(y_test,y_pred))


# In[ ]:


test=

