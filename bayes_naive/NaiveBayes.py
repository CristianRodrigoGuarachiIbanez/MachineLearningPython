"""
Created on Th Nov 17 2020
@author: Cristian Rodrigo Guarachi Ibanez
Naive Bayes
"""


from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from typing import Any, List, Tuple
training: pd.read_csv = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_train.csv')
test: pd.read_csv = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_test.csv')
df: pd.read_csv = pd.read_csv('NBdataset.csv')

print(training.head())
print(test.head())
print(df.head())

# Crear el X, Y, Training and Test
xtrain: pd.DataFrame = training.drop('Species', axis=1)
ytrain = training.loc[:, 'Species']
xtest: pd.DataFrame = test.drop('Species', axis=1)
ytest = test.loc[:, 'Species']

# Init Gaussian Classifier
model: GaussianNB = GaussianNB()

# Entrenar el model
model.fit(xtrain, ytrain)

# Predecir el Output
pred: GaussianNB = model.predict(xtest)

print(pred)

# Plotear la Confusion Matrix
mat: Any = confusion_matrix(pred, ytest)
names: Any = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()