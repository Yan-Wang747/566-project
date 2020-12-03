from operator import sub
import numpy as np
import matplotlib.pyplot as plt
import random
import shared

from sklearn.metrics import plot_confusion_matrix

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


subject = random.choice(shared.SUBJECTS)
# trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(splitMode=shared.SPLIT_MODE_BY_SUBJECT, validationRatio=0, testRatio=0.4, flatten=True, normalize=True, denoise_n=10)
trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(validationRatio=0, testRatio=0.4, flatten=True, normalize=True, denoise_n=10)
# trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(subjects=[subject], validationRatio=0, testRatio=0.4, flatten=True, normalize=True, denoise_n=10)

# y_pred = predict(trainingX, trainingLabels, testX, 2)

k = 1
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(trainingX, trainingLabels)
y_pred = classifier.predict(testX)

print(classification_report(y_pred, testLabels))
plot_confusion_matrix(classifier, testX, testLabels)

plt.show()

