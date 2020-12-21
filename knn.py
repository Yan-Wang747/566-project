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

runs = 1
ks = [1, 2, 3, 4, 5]

trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(validationRatio=0, testRatio=0.4, flatten=True)
for k in ks:
    for r in range(runs):
        print("k = {}, r = {}:".format(k, r))

        # subject = random.choice(shared.SUBJECTS)
        # trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(subjects=[subject], validationRatio=0, testRatio=0.4, flatten=True)

        classifier = KNeighborsClassifier(n_neighbors = k)
        classifier.fit(trainingX, trainingLabels)
        y_pred = classifier.predict(testX)
        
        # report = open("knn_report_ind.txt", "a")
        report = open("knn_report_rand.txt", "a")
        report.write("k = {}, r = {}:\n".format(k, r))
        report.write(classification_report(y_pred, testLabels))
        report.write('\n')
        report.close()

# plot_confusion_matrix(classifier, testX, testLabels)

# plt.show()

