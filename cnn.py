import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import shared
from preprocessing import loadData
from sklearn.metrics import classification_report

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

        '''
        nn.init.orthogonal_(self.conv1.weight.data,gain=2**0.5)
        nn.init.orthogonal_(self.conv2.weight.data,gain=2**0.5)
        nn.init.orthogonal_(self.conv3.weight.data,gain=2**0.5)
        nn.init.constant_(self.conv1.bias.data,0.0)
        nn.init.constant_(self.conv2.bias.data,0.0)
        nn.init.constant_(self.conv3.bias.data,0.0)

        nn.init.orthogonal_(self.fc1.weight.data,gain=2**0.5)
        nn.init.constant_(self.fc1.bias.data,0.0)
        nn.init.orthogonal_(self.fc2.weight.data,gain=2**0.5)
        nn.init.constant_(self.fc2.bias.data,0.0)
        nn.init.orthogonal_(self.fc3.weight.data,gain=2**0.5)
        nn.init.constant_(self.fc3.bias.data,0.0)
        nn.init.orthogonal_(self.fc4.weight.data,gain=2**0.5)
        nn.init.constant_(self.fc4.bias.data,0.0)
        nn.init.orthogonal_(self.out.weight.data,gain=2**0.5)
        nn.init.constant_(self.out.bias.data,0.0)
        '''

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

cnnModel = CnnModel()
cnnModel.cuda()

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
optimizer = optim.Adam(cnnModel.parameters(), lr=0.01)

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
    
    with torch.no_grad():
        cnnModel.eval()

        logits = cnnModel(validationX)
        _, valPredicts = torch.max(logits, axis=1)
        valPredicts = valPredicts.cpu().numpy()
        print(classification_report(valPredicts, validationLabels))
    

print('Finished Training')
