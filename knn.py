import numpy as np

from statistics import mode
from preprocessing import loadData
from sklearn.metrics import classification_report

def predict(trainingSamples, trainingLabels, testSamples, k):
    predictedLabels = []
    counter = 0
    for unseen in testSamples:
        diffMatrix = trainingSamples - unseen
        l2Distances = np.sum(np.square(diffMatrix), axis=1)
        idx = np.argpartition(l2Distances, k)[:k]
        labels = trainingLabels[idx]
        predictedLabels.append(mode(labels))

        if( counter % 100 == 0):
            print(len(predictedLabels)/len(testSamples))
        counter += 1

    return predictedLabels

trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(validationRatio=0, testRatio=0.2, flatten=True)

y_pred = predict(trainingX, trainingLabels, testX, 2)

print(classification_report(testLabels, y_pred))
