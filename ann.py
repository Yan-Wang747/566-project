import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import shared
import random
import numpy as np
from preprocessing import loadData
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

class AnnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(shared.NUM_OF_INTERP_POINTS*3, 128)
        self.fc2 = nn.Linear(128, 64)

        self.out = nn.Linear(64, 26)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.out(x)

        return logits

mode = shared.SPLIT_MODE_CLASSIC

if mode == shared.SPLIT_MODE_CLASSIC:
    trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(flatten=True, denoise_n=1)

    trainingX = torch.from_numpy(trainingX).cuda()
    trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

    validationX = torch.from_numpy(validationX).cuda()

    testX = torch.from_numpy(testX).cuda()

    trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

    reportName = "ann_report_rand_no_denoise.txt"

elif mode == shared.SPLIT_MODE_BY_SUBJECT:
    reportName = "ann_report_ind.txt"

report = open(reportName, "w")
report.close()

runs = 1
BATCH_SIZE = 500
MAX_ITER = 400
criterion = nn.CrossEntropyLoss()
for r in range(runs):
    print("r = " + str(r))

    if mode == shared.SPLIT_MODE_BY_SUBJECT:
        subject = random.choice(shared.SUBJECTS)
        trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(subjects=[subject], flatten=True)

        trainingX = torch.from_numpy(trainingX).cuda()
        trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

        validationX = torch.from_numpy(validationX).cuda()

        testX = torch.from_numpy(testX).cuda()

        trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

    trainLoader = DataLoader(trainingDataset, BATCH_SIZE, True)
    
    annModel = AnnModel()
    annModel.cuda()
    optimizer = optim.AdamW(annModel.parameters(), weight_decay=0.01)

    losses = []
    accs = []
    best_val_acc = -1
    best_epoch = None
    for epoch in range(MAX_ITER):  # loop over the dataset multiple times
        running_loss = 0.0
        annModel.train()

        for i, data in enumerate(trainLoader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits = annModel(inputs)
            loss = criterion(logits, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        losses.append(running_loss)

        with torch.no_grad():
            annModel.eval()

            logits = annModel(validationX)
            _, predicts = torch.max(logits, axis=1)
            predicts = predicts.cpu().numpy()
            acc = np.mean(predicts == validationLabels)
            accs.append(acc)
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(annModel.state_dict(), './best_ann_model.pth')
                best_epoch = epoch
        
    bestModel = AnnModel()
    bestModel.load_state_dict(torch.load('best_ann_model.pth'))
    bestModel.to('cuda')
    bestModel.eval()

    with torch.no_grad():
        logits = bestModel(testX)
        _, predicts = torch.max(logits, axis=1)
        predicts = predicts.cpu().numpy()
        
        report = open(reportName, "a")
        report.write("r = {}:\n".format(r))
        report.write(classification_report(predicts, testLabels))
        report.write('\n')
        report.close()

print('Finished Training')

'''
plt.figure()
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss')

plt.figure()
plt.plot(accs)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
'''