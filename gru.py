import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import shared
from preprocessing import loadData
from sklearn.metrics import classification_report

class GruModel(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(3, hidden_dim, n_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_dim*2, 26)


    def forward(self, x):
        out, _ = self.gru(x)

        logits = self.fc(out[:,-1,:])

        return logits

gruModel = GruModel(100, 5)
gruModel.cuda()

trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(['albert'], augmentProp=4, validationRatio=0.1, testRatio=0.1, flatten=False, denoise_n=8)

trainingX = torch.from_numpy(trainingX).cuda()
trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

validationX = torch.from_numpy(validationX).cuda()

testX = torch.from_numpy(testX).cuda()

trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

BATCH_SIZE = 250
MAX_ITER = 250

trainLoader = DataLoader(trainingDataset, BATCH_SIZE, True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(gruModel.parameters(), weight_decay=0.01)

for epoch in range(MAX_ITER):  # loop over the dataset multiple times
    running_loss = 0.0
    gruModel.train()

    for i, data in enumerate(trainLoader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits = gruModel(inputs)
        loss = criterion(logits, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        gruModel.eval()

        logits = gruModel(validationX)
        _, valPredicts = torch.max(logits, axis=1)
        valPredicts = valPredicts.cpu().numpy()
        print(classification_report(valPredicts, validationLabels))
    

print('Finished Training')
