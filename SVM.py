from sklearn import svm
import numpy as np
from preprocessing import loadData
import shared
from sklearn.metrics import classification_report
import pandas as pd



#trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(splitMode=shared.SPLIT_MODE_BY_SUBJECT, validationRatio=0, testRatio=0.2, flatten=True)
trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(validationRatio=0, testRatio=0.2, flatten=True, normalize=True)
trainx = np.array(trainingX)
trainy = np.array(trainingLabels)
testx = np.array(testX)
testy = np.array(testLabels)
kernels = ['poly'] #linear,poly,rbf,sigmoid，found rbf best
for kernel in kernels:
    clf = svm.SVC(kernel=kernel, gamma='auto', decision_function_shape="ovr", max_iter=25000)
    clf.fit(trainx, trainy)
    pred = clf.predict(testx)
    print(classification_report(testLabels, pred))


