from sklearn import svm
import numpy as np
from preprocessing import loadData
from sklearn.metrics import classification_report
import random
import shared

#subject = random.choice(shared.SUBJECTS)
trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(validationRatio=0.2, testRatio=0.2, flatten=True, normalize=True, denoise_n=10)
#trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(subjects=[subject], validationRatio=0.2, testRatio=0.2, flatten=True, normalize=True, denoise_n=10)
trainx = np.array(trainingX)
trainy = np.array(trainingLabels)
testx = np.array(testX)
testy = np.array(testLabels)
kernels = ['rbf'] #linear,poly,rbf,sigmoid，found rbf best
for kernel in kernels:
    clf = svm.SVC(kernel=kernel, gamma='auto', decision_function_shape="ovr", max_iter=25000)
    clf.fit(trainx, trainy)
    pred = clf.predict(testx)
    print(classification_report(testLabels, pred))


