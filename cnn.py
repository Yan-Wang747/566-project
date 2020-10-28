import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import shared
from preprocessing import loadData
from sklearn.metrics import classification_report

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )  

        # 16 channel, num_feature
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        # 16 channel, num_feature

        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        # 32 channel, num_feature
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)
        # 32 channel, num_feature
        
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)

        self.fc1 = nn.Linear(shared.NUM_OF_INTERP_POINTS * 64, 3200)
        self.fc2 = nn.Linear(3200, 1600)
        self.fc3 = nn.Linear(1600, 500)
        self.out = nn.Linear(500, 26)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, shared.NUM_OF_INTERP_POINTS * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

net = Net()
net.cuda()

trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(augmentProp=4, validationRatio=0.2, testRatio=0.2, flatten=False)

# N, L, C -> N, C, L
trainingX = trainingX.transpose(0, 2, 1)
validationX = validationX.transpose(0, 2, 1)
testX = testX.transpose(0, 2, 1)

trainingX = torch.from_numpy(trainingX).cuda()
trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

validationX = torch.from_numpy(validationX).cuda()

testX = torch.from_numpy(testX).cuda()

trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

BATCH_SIZE = 500
MAX_ITER = 20

trainLoader = DataLoader(trainingDataset, BATCH_SIZE, True)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=0.001)

for epoch in range(MAX_ITER):  # loop over the dataset multiple times
    running_loss = 0.0
    
    for i, data in enumerate(trainLoader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        energies = net(inputs)
        loss = criterion(energies, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        valEnergies = net(validationX)
        _, valPredicts = torch.max(valEnergies, 1)
        valPredicts = valPredicts.cpu().numpy()
        print(print(classification_report(valPredicts, validationLabels)))
    

print('Finished Training')