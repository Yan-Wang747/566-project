import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import shared
from preprocessing import loadData
from sklearn.metrics import classification_report

class LstmModel(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        '''
        self.W = nn.Parameter(torch.Tensor(3, hidden_sz*4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        '''

        self.lstm = nn.LSTM(3, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 26, bias = True)

        '''
        nn.init.orthogonal_(self.W.data, gain=2**0.5)
        nn.init.orthogonal_(self.U.data, gain=2**0.5)
        nn.init.zeros_(self.bias)
        '''

    def forward(self, x):

        init_h = torch.zeros(5, x.shape[0], self.hidden_dim).cuda()
        init_c = torch.zeros(5, x.shape[0], self.hidden_dim).cuda()
        
        '''
        for t in range(seq_sz):
            x_t = x[:, t, :]

            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :self.hidden_dim]), # input
                torch.sigmoid(gates[:, self.hidden_dim:self.hidden_dim*2]), # forget
                torch.tanh(gates[:, self.hidden_dim*2:self.hidden_dim*3]),
                torch.sigmoid(gates[:, self.hidden_dim*3:]), # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
        '''
        out, _ = self.lstm(x, (init_h, init_c))

        logits = self.fc(out[:,-1,:])

        return logits

lstmModel = LstmModel(100, 5)
lstmModel.cuda()

trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData( augmentProp=4, validationRatio=0.1, testRatio=0.1, flatten=False)

trainingX = torch.from_numpy(trainingX).cuda()
trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

validationX = torch.from_numpy(validationX).cuda()

testX = torch.from_numpy(testX).cuda()

trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

BATCH_SIZE = 250
MAX_ITER = 250

trainLoader = DataLoader(trainingDataset, BATCH_SIZE, True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(lstmModel.parameters(), weight_decay=0.01)

for epoch in range(MAX_ITER):  # loop over the dataset multiple times
    running_loss = 0.0
    lstmModel.train()

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
    
    with torch.no_grad():
        lstmModel.eval()

        logits = lstmModel(validationX)
        _, valPredicts = torch.max(logits, axis=1)
        valPredicts = valPredicts.cpu().numpy()
        print(classification_report(valPredicts, validationLabels))
    

print('Finished Training')
