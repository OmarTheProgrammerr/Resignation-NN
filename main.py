import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only display error messages
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


print(tf.__version__)

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Creating a new instance of an empty ANN (Artificial Neural Network)
ann = tf.keras.models.Sequential()
#We are using Rectifier activation function.
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Compiling the ANN. The loss here represents the cost where the "binary_crossentropy represents a common cost function."
#he optimizer's purpose is to make the model's predictions as accurate as possible by iteratively adjusting the model's parameters.
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Ok now it's time to train the ANN on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

print(X)
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
print(ann.predict(X_train)) # Use X_train instead of y_train
print(ann.predict(X_train) > 0.5) # Use ann.predict(X_train) instead of ann.preidct(y_train > 0.5)

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))