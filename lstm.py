import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from preprocessing import loadData
from sklearn.metrics import classification_report
import random
import shared
import numpy as np

import matplotlib.pyplot as plt

class LstmModel(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(3, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 26)


    def forward(self, x):
        out, _ = self.lstm(x)

        logits = self.fc(out[:,-1,:])

        return logits

mode = shared.SPLIT_MODE_CLASSIC

if mode == shared.SPLIT_MODE_CLASSIC:
    trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(denoise_n=1)

    trainingX = torch.from_numpy(trainingX).cuda()
    trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

    validationX = torch.from_numpy(validationX).cuda()

    testX = torch.from_numpy(testX).cuda()

    trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

    reportName = "lstm_report_rand_no_denoise.txt"

elif mode == shared.SPLIT_MODE_BY_SUBJECT:
    reportName = "lstm_report_ind.txt"

report = open(reportName, "w")
report.close()

runs = 1
BATCH_SIZE = 250
MAX_ITER = 400
criterion = nn.CrossEntropyLoss()
for r in range(runs):
    print("r = " + str(r))

    if mode == shared.SPLIT_MODE_BY_SUBJECT:
        subject = random.choice(shared.SUBJECTS)
        trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(subjects=[subject])

        trainingX = torch.from_numpy(trainingX).cuda()
        trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

        validationX = torch.from_numpy(validationX).cuda()

        testX = torch.from_numpy(testX).cuda()

        trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

    trainLoader = DataLoader(trainingDataset, BATCH_SIZE, True)

    lstmModel = LstmModel(100, 5)
    lstmModel.cuda()
    optimizer = optim.AdamW(lstmModel.parameters(), weight_decay=0.01)

    losses = []
    accs = []
    best_val_acc = -1
    best_epoch = None
    for epoch in range(MAX_ITER):  # loop over the dataset multiple times
        running_loss = 0.0
        lstmModel.train()
        print("epoch: " + str(epoch))
        for i, data in enumerate(trainLoader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits = lstmModel(inputs)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        losses.append(running_loss)
        with torch.no_grad():
            lstmModel.eval()

            logits = lstmModel(validationX)
            _, predicts = torch.max(logits, axis=1)
            predicts = predicts.cpu().numpy()
            acc = np.mean(predicts == validationLabels)
            accs.append(acc)
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(lstmModel.state_dict(), './best_lstm_model.pth')
                best_epoch = epoch
    
    print(best_epoch)
    print(best_val_acc)

    bestModel = LstmModel(100, 5)
    bestModel.load_state_dict(torch.load('best_lstm_model.pth'))
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