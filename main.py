import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only display error messages
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


print(tf.__version__)

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print(X)