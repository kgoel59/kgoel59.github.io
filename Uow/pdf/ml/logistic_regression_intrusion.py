import numpy as np
import pandas as pd
import warnings

# Disable warnings for cleaner output during execution
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from ucimlrepo import fetch_ucirepo

# Setting a constant seed for reproducibility
random_state = 42

# Function to update target labels for binary classification
def update_labels(targets):
    labels = list(targets['Attack_type'].unique())[3:]
    y = []
    for i in range(len(targets)):
        if targets['Attack_type'][i] in labels:
            y.append(1)
        else:
            y.append(0)
    return pd.DataFrame(y, columns=['Attack_type'])

# Function to fetch and prepare the dataset
def get_dataset():
    print("Fetching Dataset RT_IOT2022 from ucirepo")
    rt_iot2022 = fetch_ucirepo(id=942)
    X = rt_iot2022.data.features
    y = update_labels(rt_iot2022.data.targets)
    return X, y

# Custom transformer for encoding categorical features
class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['proto']]).toarray()
        column_names = ["tcp", "udp", "icmp"]
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        return X

# Custom transformer for dropping unnecessary features
class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(["proto", "id.orig_p", "id.resp_p", "service"], axis=1, errors="ignore")

preprocess = Pipeline(steps=[
    ("encoder", FeatureEncoder()),
    ("dropper", FeatureDropper()),
    ("scaler", StandardScaler()),
])

# Function to create logistic regression models with different penalties and solvers
def get_models():
    print("Creating Models")
    models = dict()
    penalties = {
        None: ["lbfgs","newton-cg","saga"],
        "l1": ["saga"],
        "l2": ["lbfgs","newton-cg","saga"],
        "elasticnet": ["saga"]
    }
    l1_ratio = 0.5
    for k in penalties:
        for s in penalties[k]:
            if k == "elasticnet":
                model = LogisticRegression(random_state=random_state, penalty=k, solver=s, l1_ratio=l1_ratio)
            else:
                model = LogisticRegression(random_state=random_state, penalty=k, solver=s)
            steps = [
                ("preprocess", preprocess),
                ("model", model)
            ]
            models[f"{k}|{s}"] = Pipeline(steps=steps)
    return models

# Function to train and evaluate the logistic regression models
def train_evaluate(X, y, model, solver, penalty):
    kf = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
    for train_indices, test_indices in kf.split(X, y[["Attack_type"]].join(X["proto"])):
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]

        X_train = model['preprocess'].fit_transform(X_train)
        y_train = y_train.to_numpy()
        X_test = model['preprocess'].fit_transform(X_test)
        y_test = y_test.to_numpy()

        print(f"Training model with {penalty} penalty and {solver} solver")
        model["model"].fit(X_train, y_train)
        y_pred = model["model"].predict(X_test)

        return (
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)
        )

# Main function for running experiment 1
def experiment1(X,y):
    print("Running Experiment 1")
    print("Solver and Regularization Pairings")
    models = get_models()
    results = []
    print("Training and Evaluating Models")
    for name, model in models.items():
        solver = name.split("|")[1]
        penalty = name.split("|")[0]
        accuracy, precision, recall, f1 = train_evaluate(X, y, model, solver, penalty)
        results.append([penalty, solver, accuracy, precision, recall, f1])

    results = pd.DataFrame(results, columns=["penalty", "solver", "accuracy", "precision", "recall", "f1"])
    print(results.round(4))
    print(f"max: {results[results['accuracy'] == results['accuracy'].max()]}")


# Functions for PCA-based model creation and evaluation
def get_models_pca(start,end):
  print("Creating Models PCA")
  models = dict()
  for i in range(start,end):
    steps = [
      ("preprocess",preprocess),
      ('pca', PCA(n_components=i)),
      ('model', LogisticRegression(penalty='l2',solver='lbfgs',max_iter=1000000))]
    models[i] = Pipeline(steps=steps)
  return models

def train_evaluate_pca(model, X, y):
    kf = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=random_state)
    cv = kf.split(X, y[["Attack_type"]].join(X["proto"]))
    accuracy = cross_val_score(model, X, y.to_numpy().ravel(), scoring='accuracy', cv=cv, n_jobs=-1)

    return accuracy

# Main function for running experiment 2
def experiment2(X,y):
    print("Running Experiment 2")
    print("Feature Selection via PCA")
    models = get_models_pca(55,65)
    results, names = list(), list()
    print("Training and Evaluating Model PCA")
    for name, model in models.items():
        print(f"Dimension Size - {name}")
        scores = train_evaluate_pca(model, X, y)
        results.append(scores)
        names.append(name)
    
    results = pd.DataFrame(zip([result[0] for result in results],names),columns=['accuracy','dimension'])
    print(f"max: {results[results['accuracy'] == results['accuracy'].max()]}")

# Main code entry point
print("----------------------------")
print("Karan Goel - 7836685")
print("Machine Learning")
print("----------------------------")

X, y = get_dataset()
print("----------------------------")
experiment1(X,y)
print()
print("----------------------------")
print()
experiment2(X,y)
print("----------------------------")
