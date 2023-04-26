#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd 
import numpy as np
from scipy.stats import randint
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import cross_val_score


# In[2]:


boston = load_boston()


# In[4]:


X = boston.data
X.shape


# In[6]:


y = boston.target
y.shape


# In[7]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[15]:


print(f'X train= {X_train.shape}')
print(f'y train= {y_train.shape}')
print(f'X test = {X_test.shape}')
print(f'y test = {y_test.shape}')


# In[19]:


forest_reg = RandomForestRegressor()


# In[25]:


forest_reg.fit(X_train,y_train)


# In[26]:


#Calculating Details
print('Random Forest Regressor Train Score is : ' , forest_reg.score(X_train, y_train))
print('Random Forest Regressor Test Score is : ' , forest_reg.score(X_test, y_test))


# In[24]:


forest_scores = cross_val_score(forest_reg,X_train, y_train,scoring = "neg_mean_squared_error", cv = 10)
forest_rmse_scores = np.sqrt(-forest_scores)


# In[27]:


print("Scores: ", forest_rmse_scores)
print("Mean: ", forest_rmse_scores.mean())
print("Standard Deviation: ", forest_rmse_scores.std())


# In[29]:


# Define the hyperparameter space
param_distribs = {
        'n_estimators': np.random.randint(1, 200, 10),
        'max_features': np.random.randint(1, 8, 10),
    }


# In[30]:


rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)


# In[31]:


rnd_search.fit(X_train,y_train)


# In[32]:


# Print the best hyperparameters
print("Best Hyperparameters:", rnd_search.best_params_)


# In[33]:


cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[34]:


final_model = rnd_search.best_estimator_
final_predictions = final_model.predict(X_test)


# In[41]:


final_mse = mean_squared_error(y_test, final_predictions)


# In[42]:


final_rmse = np.sqrt(final_mse)
final_rmse


# In[43]:


3.6223447056674396 - 3.15347816881884


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




