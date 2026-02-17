"""
Finding common symptoms and predictors for heart disease across all sexes in the dataset.
Model types used:
-
-


@author: Lauren Hendley
@author: Sayohn David
"""



'''
1. Investigation- look @ dataset, find the parameters
2. Model development - model fitting, train/test split, model prediction
    - Use different models (random forests, support vector machines, SCARF, etc)
3. Find model results
4. Compare accuracies of the different models (see which is most accurate)
'''




'''
Methods required:
1. Main method
- Train/test split for each model
2. Dataset breakdown
3. Data preprocessing
4. Comparison of models (AUC, acccuracy, precision, recall, F1 score)

1. Logistic Regression
2. L1/L2/ElasticNet
3. Random Forest
4. XGBoost
'''



# IMPORTS
import pandas as pd
import numpy as np
import matplotlib as plt
import xgboost as xgb

from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


## Evaluate markers in the dataset
def dataset_eval(ds):
    ds.head()

    print("Dataset shape: ", ds.shape)

    print("\nDataset information: ", ds.info())

    print("\nDataset description: ", ds.describe())


## XGBoost implementation
def exgee(x_train, y_train, x_test, y_test):
    print("Starting XGBoost...")

    params = {
        "learning_rate" : [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "max_depth" : [3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight" : [1, 3, 5, 7],
        "gamma" : [0.0, 0.1, 0.2, 0.3, 0.4],
        "colsample_bytree" : [0.3, 0.4, 0.5, 0.7]
    }

    classifier = XGBClassifier()
    random_search = RandomizedSearchCV(estimator = XGBClassifier(), param_distributions = params, n_iter = 5, 
                                       scoring = 'roc_auc', n_jobs = -1, cv = 5, verbose = 3)
    
    random_search.fit(x_train, y_train)

    classifier = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.2, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=6,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

    score = cross_val_score(classifier, x, y, cv = 10)
    score.mean()
    classifier.fit(x_train, y_train)

    pred = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


def ran_forest(x_train, y_train, x_test, y_test):
    # Number of trees in random forest
    n_estimators=[20,60,100,120]

    # Number of features to consider at every split
    max_features=[0.2,0.6,1.0]

    # Maximum number of levels in tree
    max_depth=[2,4,8,None]

    #Number of samples
    max_samples=[0.5,0.75,1.0]

    param_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth':max_depth,
        'max_samples': max_samples
    }

    rf = RandomForestClassifier(oob_score = True)
    rf_grid = GridSearchCV(estimator = rf, param_grid = param_grid,
                           cv = 5, verbose = 2, n_jobs = -1)
    
    rf_grid.fit(x_train, y_train)

    rf_grid.best_params_

    rf.oob_score_

    rf_grid.best_score_

    y_pred = rf.predict(x_test)
    accuracy_score(y_test, y_pred)



# Main method
if __name__ == "__main__":
    ds = pd.read_csv("/kaggle/input/heart-disease-dataset/heart.csv")
    dataset_eval(ds)

    features = [feature for feature in ds.columns if len(ds[feature].unique()) < 10]

    tmp_ds = pd.get_dummies(ds, columns = features)
    std_scl = StandardScaler()
    tmp_ds[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = std_scl.fit_transform([['age', 'trestbps', 'chol', ' thalach', 'oldpeak']])

    x = tmp_ds.drop(["target"], axis = 1)
    y = tmp_ds.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    exgee(x_train, y_train, x_test, y_test)

    ran_forest(x_train, y_train, x_test, y_test)