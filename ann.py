import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import shared
from preprocessing import loadData
from sklearn.metrics import classification_report

class AnnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(shared.NUM_OF_INTERP_POINTS*3, 128)
        self.fc2 = nn.Linear(128, 64)

        self.out = nn.Linear(64, 26)

        '''
        nn.init.orthogonal_(self.fc1.weight.data,gain=2**0.5)
        nn.init.constant_(self.fc1.bias.data,0.0)
        nn.init.orthogonal_(self.out.weight.data,gain=2**0.5)
        nn.init.constant_(self.out.bias.data,0.0)
        '''

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.out(x)

        return logits

annModel = AnnModel()
annModel.cuda()

trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(augmentProp=4, validationRatio=0.2, testRatio=0.2, flatten=True)

trainingX = torch.from_numpy(trainingX).cuda()
trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

validationX = torch.from_numpy(validationX).cuda()

testX = torch.from_numpy(testX).cuda()

trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

BATCH_SIZE = 500
MAX_ITER = 20

trainLoader = DataLoader(trainingDataset, BATCH_SIZE, True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(annModel.parameters(), weight_decay=0.01)

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
    
    with torch.no_grad():
        annModel.eval()

        logits = annModel(validationX)
        _, valPredicts = torch.max(logits, axis=1)
        valPredicts = valPredicts.cpu().numpy()
        print(classification_report(valPredicts, validationLabels))
    

print('Finished Training')
