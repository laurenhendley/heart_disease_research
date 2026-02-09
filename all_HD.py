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
from sklearn.metrics import accuracy_score, recall_score


## Evaluate markers in the dataset
def dataset_eval(ds):
    ds.head()

    print("Dataset shape: ", ds.shape)

    print("\nDataset information: ", ds.info())

    print("\nDataset description: ", ds.describe())


# Main method
if __name__ == 'main':
    ds = pd.read_csv("/kaggle/input/heart-disease-dataset/heart.csv")
    dataset_eval(ds)