from sklearn.naive_bayes import GaussianNB
import numpy as np
import shared
import pandas as pd
from preprocessing import loadData
from sklearn.metrics import classification_report
from sklearn import metrics
import torch
# from keras.utils import to_categorical

# trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(splitMode=shared.SPLIT_MODE_BY_SUBJECT, validationRatio=0, testRatio=0.2, flatten=True)
trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(['albert'], validationRatio=0, testRatio=0.2,
                                                                                       flatten=True, normalize=True)

# trainingX = to_categorical(trainingX)
# testX = to_categorical(testX)

n_y = np.zeros(26)
for i in range(26):
    n_y[i] = (trainingLabels == i).sum()
P_y = n_y / n_y.sum()  # p(y)
n_x = np.zeros((26, 300))
for j in range(26):
    n_x[j] = np.array(trainingX[trainingLabels == j].sum(axis=0))
P_xy = n_x / n_y.reshape(26, 1)  # P(x|y)


def bayes_pred(x):
    # x = np.expand_dims(x, axis=0)
    p_xy = P_xy * x + (1 - P_xy) * (1 - x)
    p_xy = p_xy.reshape(26, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y


y_pred = bayes_pred(testX)
print(classification_report(testLabels, y_pred))

# Using Gaussian Classifier
# model = GaussianNB()
# model.fit(trainingX, trainingLabels)
# y_pred = model.fit(trainingX, trainingLabels).predict(testX)
# metrics.accuracy_score(testLabels,y_pred)
