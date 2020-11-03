import torch
from torch import optim
import torch.nn as nn
import torch.optim
import numpy as np

from preprocessing import loadData
from sklearn.metrics import classification_report

trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(validationRatio=0.2, testRatio=0.2, flatten=True, normalize=True)

trainingX = torch.from_numpy(trainingX).cuda()
# trainingLabels = torch.from_numpy(trainingLabels).long().cuda()
trainingLabelsOneHot = []
for i, l in enumerate(trainingLabels):
    trainingLabelsOneHot.append([0]*26)
    trainingLabelsOneHot[i][l] = 1

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

MAX_ITER = 20000
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

    print(loss.item())

    # loss.backward()
    logits.backward(gv)
    optimizer.step()

with torch.no_grad():
    logits = validationX @ w
    # logits = model(validationX)
    _, valPredicts = torch.max(logits, axis=1)
    valPredicts = valPredicts.cpu().numpy()
    print(classification_report(valPredicts, validationLabels))
    