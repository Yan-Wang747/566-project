import numpy as np
from preprocessing import loadData
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV


trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(validationRatio=0.2, testRatio=0.2, flatten=True, normalize=True, denoise_n=10)
trainx = np.array(trainingX[1:4000, ])
trainy = np.array(trainingLabels[1:4000, ])
#trainx = np.array(trainingX)
#trainy = np.array(trainingLabels)
testx = np.array(testX)
testy = np.array(testLabels)

params = [dict(kernel=['rbf'], gamma=[1, 0.1, 1e-2, 1e-3, 1e-4], C=[1, 10, 100, 1000])]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
BO_model = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv)
BO_model.fit(trainx,trainy)

# View the accuracy score
print('Best score for training data:', BO_model.best_score_, "\n")

# View the best parameters for the model found using grid search
print('Best C:', BO_model.best_estimator_.C, "\n")
print('Best Kernel:', BO_model.best_estimator_.kernel, "\n")
print('Best Gamma:', BO_model.best_estimator_.gamma, "\n")

svm_model_new = SVC(gamma=0.01, C=100, kernel='rbf', decision_function_shape="ovr")
svm_model_new.fit(trainx, trainy)
pred = BO_model.predict(testx)
print(classification_report(testy, pred))