# Core
import numpy as np
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

df = pd.read_csv('../../datasets/breast-cancer.csv')
print(df.head())
print(df.describe())

# Prep features and target
X = df.drop(['id', 'diagnosis'], axis=1).values
y = df.diagnosis.values

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost classifier base model
xgb_clf = xgb.XGBClassifier(
    n_jobs = 1,
    booster = 'gbtree',
    objective = 'binary:logistic',
    eval_metric = 'auc',
    tree_method='gpu_hist',
    enable_categorical = True,
    early_stopping_rounds = 5,
    verbosity = 0
)


