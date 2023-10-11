#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.kaggle.com/code/abhinavbhuyan/classification-xgboost-with-bayesian-optimization?scriptVersionId=93389201" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>

# ## Introduction
# We perform binary classification on breast cancer dataset using XGBoost, augmented with hyperparameter optimization using Bayesian sampling.
# - Load breast cancer dataset
#     - Cleaning & EDA
#     - Visualization
# - Feature engineering
#     - Prepare feature and targets and drop non-informative features
#     - Split into train and test sets
# - Bayesian optimization
#     - Define XGBoost classifier model, search space, evaluation metric & cross-validaion strategy (stratified k-fold)
#     - Run optimization for n iterations to find best parameters
# - Model training and analysis
#     - Configure XGBoost model with best parameters and fit to train set
#     - Draw tree graphs
#     - Plot feature importance
# - Evaluation
#     - Use fitted model to make predictions on test set
#     - Compute mean accuracy
#     - Draw confusion matrix

# ### Imports

# In[ ]:


# Core
import numpy as np
np.int = int
import pandas as pd
from time import time
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ML
import xgboost as xgb
from xgboost import XGBClassifier, DMatrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# ## Data and preprocessing

# In[2]:


df = pd.read_csv('../../datasets/breast-cancer.csv')
df.head()


# In[3]:


df.describe()


# In[4]:


# Prep features and target
X = df.drop(['id', 'diagnosis'], axis=1).values
y = df.diagnosis.values


# In[5]:


# Split into train, validation & test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
y_valid = le.fit_transform(y_valid)
eval_set = [(X_valid, y_valid)]


# ## Bayesian optimization

# In[6]:


# XGBoost classifier base model
xgb_clf = xgb.XGBClassifier(
    n_jobs = 1,
    booster = 'gbtree',
    objective = 'binary:logistic',
    eval_metric = 'auc', 
    tree_method='hist', 
    enable_categorical = True, 
    verbosity = 0
)


# In[7]:


# Define search space
search_spaces = {
     'learning_rate': Real(0.01, 1.0, 'log-uniform'),
     'max_depth': Integer(2, 20),
     'min-child-weight': Integer(1, 5),
     'reg_lambda': Real(1e-9, 100., 'log-uniform'),
     'reg_alpha': Real(1e-9, 100., 'log-uniform'),
     'gamma': Real(1e-9, 0.5, 'log-uniform'),  
     'n_estimators': Integer(10, 5000)
}


# In[8]:


# Create Bayesian CV for HP optimization
bayes_cv = BayesSearchCV(
                    estimator = xgb_clf,                                    
                    search_spaces = search_spaces,                      
                    scoring = 'roc_auc',                                  
                    cv = StratifiedKFold(n_splits=5, shuffle=True),                                   
                    n_iter = 20,                                      
                    n_points = 5,                                       
                    n_jobs = 1,                                         
                    iid = False,                                        
                    refit=False,
                    verbose = 1,
                    random_state=42
)                               


# In[9]:


# Run bayesian CV

y_train = le.fit_transform(y_train)

get_ipython().run_line_magic('time', 'bayes_cv.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=5, verbose=True)')


# In[10]:


# Show best params
print('Best parameters:')
pprint.pprint(bayes_cv.best_params_)
print("Best score = %.3f after %d runs" % (bayes_cv.best_score_, bayes_cv.n_iter))


# ## Model training

# In[11]:


# Create XGBoost classifier with best params
xgb_clf_tuned = xgb.XGBClassifier(
        n_jobs = 5,
        objective = 'binary:logistic',
        eval_metric = 'auc', 
        tree_method = 'hist', 
        booster = 'gbtree',
        enable_categorical = True, 
        **bayes_cv.best_params_
)


# In[14]:


# Fit to train data
xgb_clf_tuned.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=5, verbose=True)


# In[18]:


# Visualize decision tree
xgb.to_graphviz(xgb_clf_tuned, num_tree=10)


# In[19]:


# Feature importance plot
fig, ax = plt.subplots(figsize = (10, 8))
xgb.plot_importance(xgb_clf_tuned, ax=ax)


# ## Prediction and evaluation

# In[24]:


# Predict on test set using fitted model
y_pred = xgb_clf_tuned.predict(X_test)
print(y_pred)
print(y_test)


# In[25]:


# Mean accuracy
y_test_num = np.where(y_test == 'M', 1, 0)
accuracy_score(y_test_num, y_pred)


# In[27]:


# Confusion matrix
conf = confusion_matrix(y_test_num, y_pred)
df_conf = pd.DataFrame(conf, index = ['B', 'M'], columns = ['B', 'M'])
plt.figure(figsize = (6, 6))
sns.heatmap(df_conf, annot=True, cmap='viridis')


# In[ ]:




