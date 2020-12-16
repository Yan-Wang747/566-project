import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import shared
from preprocessing import loadData
from sklearn.metrics import classification_report
import numpy as np
import random
import matplotlib.pyplot as plt

class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2
        )  
        # 16 channel, num_feature
        
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # 16 channel, num_feature/2

        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        # 32 channel, num_feature/2

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # 32 channel, num_feature/4
        
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )
        # 64 channel, num_feature/4

        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        # 64 channel, num_feature/20

        self.fc1 = nn.Linear(shared.NUM_OF_INTERP_POINTS * 64 // 20, 320)
        #self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(320, 160)
        #self.drop2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(160, 80)
        #self.drop3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(80, 40)
        #self.drop4 = nn.Dropout(p=0.2)
        self.out = nn.Linear(40, 26)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, shared.NUM_OF_INTERP_POINTS * 64 // 20)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)
        return x

mode = shared.SPLIT_MODE_CLASSIC

if mode == shared.SPLIT_MODE_CLASSIC:
    trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(denoise_n=1)

    # N, L, C -> N, C, L
    trainingX = trainingX.transpose(0, 2, 1)
    validationX = validationX.transpose(0, 2, 1)
    testX = testX.transpose(0, 2, 1)

    trainingX = torch.from_numpy(trainingX).cuda()
    trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

    validationX = torch.from_numpy(validationX).cuda()

    testX = torch.from_numpy(testX).cuda()

    trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

    reportName = "cnn_report_rand_denoise.txt"

elif mode == shared.SPLIT_MODE_BY_SUBJECT:
    reportName = "cnn_report_ind.txt"

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
        trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(subjects=[subject])

        # N, L, C -> N, C, L
        trainingX = trainingX.transpose(0, 2, 1)
        validationX = validationX.transpose(0, 2, 1)
        testX = testX.transpose(0, 2, 1)

        trainingX = torch.from_numpy(trainingX).cuda()
        trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

        validationX = torch.from_numpy(validationX).cuda()

        testX = torch.from_numpy(testX).cuda()

        trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

    trainLoader = DataLoader(trainingDataset, BATCH_SIZE, True)

    cnnModel = CnnModel()
    cnnModel.cuda()
    optimizer = optim.AdamW(cnnModel.parameters(), weight_decay=0.01)

    losses = []
    accs = []
    best_val_acc = -1
    best_epoch = None
    for epoch in range(MAX_ITER):  # loop over the dataset multiple times
        running_loss = 0.0
        cnnModel.train()

        for i, data in enumerate(trainLoader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits = cnnModel(inputs)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        losses.append(running_loss)

        with torch.no_grad():
            cnnModel.eval()

            logits = cnnModel(validationX)
            _, predicts = torch.max(logits, axis=1)
            predicts = predicts.cpu().numpy()
            acc = np.mean(predicts == validationLabels)
            accs.append(acc)
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(cnnModel.state_dict(), './best_cnn_model.pth')
                best_epoch = epoch

    print(best_epoch)
    print(best_val_acc)

    bestModel = CnnModel()
    bestModel.load_state_dict(torch.load('best_cnn_model.pth'))
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