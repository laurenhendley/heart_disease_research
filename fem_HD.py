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


### PREREQS
## pip install ucimlrepo


# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from ucimlrepo import fetch_ucirepo 


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

    smote = SMOTE(random_state=42)
    x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train) 

    print("Balanced class distribution:", pd.Series(y_train_bal).value_counts())

    params = {
        "learning_rate" : [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "max_depth" : [2, 3, 4, 5, 6],
        "min_child_weight" : [1, 3, 5, 7],
        "gamma" : [0.0, 0.1, 0.2, 0.3, 0.4],
        "colsample_bytree" : [0.3, 0.4, 0.5, 0.7]
    }

    random_search = RandomizedSearchCV(estimator = XGBClassifier(), param_distributions = params, n_iter = 5, scoring = 'f1', n_jobs = -1, cv = 5, verbose = 3)
    
    random_search.fit(x_train_bal, y_train_bal)

    print("Best params:", random_search.best_params_)

    classifier = random_search.best_estimator_

    pred = classifier.predict(x_test)
    print("Predicted class distribution:", pd.Series(pred).value_counts())
    
    return classifier


def ran_forest(x_train, y_train, x_test, y_test):
    smote = SMOTE(random_state=42)
    x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train) 

    # Number of trees in random forest
    n_estimators=[20,60,100,120]

    # Number of features to consider at every split
    max_features=[0.2,0.6,1.0]

    # Maximum number of levels in tree
    max_depth=[2,4,6]

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
    
    rf_grid.fit(x_train_bal, y_train_bal)
    
    best_rf = rf_grid.best_estimator_

    print("Best parameters: ", rf_grid.best_params_) 

    print("OOB Score: ", best_rf.oob_score_) 

    print("Best score: ", rf_grid.best_score_) 

    y_pred = rf_grid.best_estimator_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(y_test, y_pred))
    print("AUC: %.3f" % roc_auc_score(y_test, y_pred))


## Logistic Regression implementation
def log_reg(x_train,y_train,x_test,y_test):
    print("Starting Logistic Regression...")

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train,y_train)

    pred = model.predict(x_test)

    accuracy = accuracy_score(y_test,pred)
    print("Accuracy: %.2f%%" % (accuracy*100.0))
    print(classification_report(y_test, pred))
    print("AUC: %.3f" % roc_auc_score(y_test, pred))


## Regularized Logistic Regression implementation (L1, L2, ElasticNet)
def l1l2(x_train,y_train,x_test,y_test):
    print("Starting Regularized Logistic Regression...")

    #L2 Ridge, L1 Lasso, and ElasticNet configurations
    configs = [
        ("L2 Ridge", {"penalty": "l2", "solver": "lbfgs",     "max_iter": 1000, "C": 1.0}),
        ("L1 Lasso", {"penalty": "l1", "solver": "liblinear", "max_iter": 1000, "C": 1.0}),
        ("ElasticNet", {"penalty": "elasticnet", "solver": "saga", "max_iter": 1000, "C": 1.0, "l1_ratio": 0.5}),
    ]

    for name, kwargs in configs:
        model = LogisticRegression(**kwargs)
        model.fit(x_train,y_train)

        pred = model.predict(x_test)

        accuracy = accuracy_score(y_test,pred)
        print("%s Accuracy: %.2f%%" % (name, accuracy*100))
        print(classification_report(y_test, pred))
        print("AUC: %.3f" % roc_auc_score(y_test, pred))
        

def ds_clean():
    heart_disease = fetch_ucirepo(id=45)
    ds = pd.concat([heart_disease.data.features, heart_disease.data.targets], axis=1)
    
    # binarize target 
    ds['num'] = (ds['num'] > 0).astype(int)
    ds = ds.rename(columns={'num': 'target'})

    ds = ds.dropna()

    ds = ds[ds["sex"] == 0]  # ONLY WOMEN

    print("Class balance:\n", ds['target'].value_counts())
    print("Class ratio:", ds['target'].value_counts(normalize=True).round(3).to_dict())

    dataset_eval(ds)

    return ds


# Main method
if __name__ == "__main__":
    ds = ds_clean()

    features = [feature for feature in ds.columns if len(ds[feature].unique()) < 10 and feature != "target"]

    tmp_ds = pd.get_dummies(ds, columns = features)
    std_scl = StandardScaler()
    tmp_ds[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = std_scl.fit_transform(tmp_ds[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

    x = tmp_ds.drop(["target"], axis = 1)
    y = tmp_ds.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    smote = SMOTE(random_state=42)
    x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)

    classifier = exgee(x_train, y_train, x_test, y_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(classifier, x, y, cv=skf, scoring='roc_auc')
    print(f"AUC: {score.mean():.3f} (+/- {score.std():.3f})")

    pred = classifier.predict(x_test)

    print("Test accuracy:  %.2f%%" % (accuracy_score(y_test, pred) * 100.0))


    accuracy = accuracy_score(y_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    ran_forest(x_train, y_train, x_test, y_test)

    log_reg(x_train_bal, y_train_bal, x_test, y_test)

    l1l2(x_train_bal, y_train_bal, x_test, y_test)