import numpy as np
from preprocessing import loadData
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(validationRatio=0, testRatio=0.2, flatten=True, normalize=True)
trainx = np.array(trainingX[1:4000,])
trainy = np.array(trainingLabels[1:4000,])
#trainx = np.array(trainingX)
#trainy = np.array(trainingLabels)
testx = np.array(testX)
testy = np.array(testLabels)

params_grid = [dict(kernel=['rbf'], gamma=[0.1, 1e-2, 1e-3, 1e-4], C=[1, 10, 100, 1000])]
svm_model = GridSearchCV(SVC(), params_grid, cv=2)
svm_model.fit(trainx, trainy)
pred = svm_model.predict(testx)
print(classification_report(testLabels, pred))

# View the accuracy score
print('Best score for training data:', svm_model.best_score_,"\n")

# View the best parameters for the model found using grid search
print('Best C:',svm_model.best_estimator_.C,"\n")
print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")