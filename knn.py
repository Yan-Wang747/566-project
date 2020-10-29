import numpy as np
import shared

from preprocessing import loadData
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier

def predict(trainingSamples, trainingLabels, testSamples, k):
    predictedLabels = []
    counter = 0
    for unseen in testSamples:
        diffMatrix = trainingSamples - unseen
        l2Distances = np.sum(np.square(diffMatrix), axis=1)
        idx = np.argpartition(l2Distances, k)[:k]
        labels = (trainingLabels[idx]).tolist()
        predictedLabels.append(max(set(labels), key = labels.count))

        if( counter % 100 == 0):
            print(len(predictedLabels)/len(testSamples))
        counter += 1

    return predictedLabels

# trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(splitMode=shared.SPLIT_MODE_BY_SUBJECT, validationRatio=0, testRatio=0.2, flatten=True)
trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(validationRatio=0, testRatio=0.2, flatten=True, normalize=True)

y_pred = predict(trainingX, trainingLabels, testX, 2)
'''
classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(trainingX, trainingLabels)
y_pred = classifier.predict(testX)
'''
print(classification_report(testLabels, y_pred))
