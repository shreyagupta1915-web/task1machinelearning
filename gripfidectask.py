#!/usr/bin/env python
# coding: utf-8

# # Name:Shreya Gupta

# # Spark Foundation Task 1: Prediction using supervised machine learning
# <b> In this task we will predict the percentage marks that a student is expected to score based upon the number of hours they studied. A simple linear regression task with the involvement of two variables.

# In[1]:


#importing libraries
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing datasets
data=pd.read_csv('http://bit.ly/w-data')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# Lets visualize the correlation between two variables.

# In[6]:


data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Scores Obtained')
plt.show()


# <b> From the graph its clear that there is a positive linear relation between number of hours studied and scores obtained and linear regression model can be used here.
# 
# 

# # Data Preparation

# In this step we will divide the whole dataset in to two parts for testing and training.

# In[7]:


X=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[8]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# # Training Algorithm

# After training and testing of data set now we will train our algorithm

# In[11]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()


# In[12]:


reg.fit(X_train,y_train)


# In[13]:


print("Training is completed")


# # Visualizing model

# In[14]:


l=reg.coef_*X+reg.intercept_

plt.scatter(X,y)
plt.plot(X,l)
plt.show()


# In[15]:


print("Intercept :")
print(reg.intercept_)


# In[16]:


print("Coefficient:")
print(reg.coef_)


# # Making Prediction

# In[18]:


y_pred=reg.predict(X_test)


# In[19]:


y_pred


# ddddd

# In[20]:


#Comparing Actual vs Predicted
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


# In[21]:


#Visualizing the predicted and actual values
plt.scatter(X_test,y_test)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Testing data actual values')
plt.show()


# In[22]:


plt.scatter(X_test,y_pred,marker='v')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Testing data predicted values')
plt.show()


# # Model Evaluation

# In[23]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Absolute Error:',metrics.mean_squared_error(y_test,y_pred))
print('Mean Absolute Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# # Thank you!
