#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


# In[2]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]:


# Define hyperparameters for Random Forest classifier
params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest classifier
rf = RandomForestClassifier(random_state=42)


# In[4]:


# Use RandomizedSearchCV to find the best combination of hyperparameters
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=100,
                               cv=5, verbose=2, random_state=42, n_jobs=-1)


# In[8]:


# Get the best hyperparameters
best_params = rf_random.best_params_
print("Best hyperparameters:", best_params)


# In[12]:


#cvres = rf_random.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #print(np.sqrt(-mean_score), params)


# In[15]:


# Fit RandomizedSearchCV to training data
# Initialize Random Forest classifier with best hyperparameters
rf_best = RandomForestClassifier(**rf_random.best_params_, random_state=42)


# In[18]:


rf_best.fit(X_train,y_train)


# In[20]:


rf_best.score(X_train, y_train)


# In[19]:


rf_best.score(X_test, y_test)


# In[21]:


scores = cross_val_score(rf_best, X_train, y_train , scoring ="neg_mean_squared_error",cv = 10)
lin_rmse_scores = np.sqrt(-scores)


# In[22]:


print("Scores: ", lin_rmse_scores)
print("Mean: ", lin_rmse_scores.mean())
print("Standard Deviation: ", lin_rmse_scores.std())


# In[27]:


# Make predictions on test data
y_pred = rf_best.predict(X_test)

# Evaluate performance on test data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-score: {f1_score(y_test, y_pred):.3f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




