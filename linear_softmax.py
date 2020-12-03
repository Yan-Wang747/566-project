import torch
from torch import optim
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
import shared
import random

from preprocessing import loadData
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

subject = random.choice(shared.SUBJECTS)
# trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(validationRatio=0.2, testRatio=0.2, flatten=True, normalize=True, denoise_n=10)
trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(subjects=[subject], validationRatio=0.2, testRatio=0.2, flatten=True, normalize=True, denoise_n=10)

trainingX = torch.from_numpy(trainingX).cuda()
# trainingLabels = torch.from_numpy(trainingLabels).long().cuda()
trainingLabelsOneHot = []
for l in trainingLabels:
    trainingLabelsOneHot.append([0]*26)
    trainingLabelsOneHot[-1][l] = 1

trainingLabelsOneHot = np.array(trainingLabelsOneHot)

trainingLabelsOneHot = torch.from_numpy(trainingLabelsOneHot).long().cuda()

validationX = torch.from_numpy(validationX).cuda()

testX = torch.from_numpy(testX).cuda()

w = torch.randn(300, 26, requires_grad=True, device="cuda")

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(300, 26)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

# model = LogisticRegression()
# model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW([w], weight_decay=0.01)

MAX_ITER = 5500
losses = []
accs = []
best_val_acc = -1
best_w = None
best_epoch = None

for epoch in range(MAX_ITER):  # loop over the dataset multiple times
    # zero the parameter gradients
    optimizer.zero_grad()

    logits = trainingX @ w
    # logits = model(trainingX)
    # loss = criterion(logits, trainingLabels)
    with torch.no_grad():
        maxima, _ = torch.max(logits, axis=1)
        logitsReduced = logits - maxima.unsqueeze(1)
        p = torch.exp(logitsReduced) / torch.sum(torch.exp(logitsReduced), axis=1).unsqueeze(1)
        gv = p - trainingLabelsOneHot
        loss = -torch.sum(torch.log(torch.sum(trainingLabelsOneHot*p, axis=1))) / len(trainingX)

    losses.append(loss.item())

    # loss.backward()
    logits.backward(gv)
    optimizer.step()

    with torch.no_grad():
        logits = validationX @ w

        _, predicts = torch.max(logits, axis=1)
        predicts = predicts.cpu().numpy()
        acc = np.mean(predicts == validationLabels)
        accs.append(acc)
        if acc > best_val_acc:
            best_val_acc = acc
            best_w = w
            best_epoch = epoch


print(best_epoch)
print(best_val_acc)
with torch.no_grad():
    logits = testX @ best_w

    _, predicts = torch.max(logits, axis=1)
    predicts = predicts.cpu().numpy()
    print(classification_report(predicts, testLabels))

plt.figure()
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss')

plt.figure()
plt.plot(accs)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
