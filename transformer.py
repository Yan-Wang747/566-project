import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import math
from preprocessing import loadData
from sklearn.metrics import classification_report

class TransformerModel(nn.Module):

    def __init__(self, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(3, dropout)
        encoder_layers = TransformerEncoderLayer(3, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.fc1 = nn.Linear(300, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 26)

        self.init_weights()
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def init_weights(self):
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)

        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)

        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)


    def forward(self, src, src_mask):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)

        output = output.transpose(0, 1)
        output = torch.reshape(output, [output.shape[0], 1, -1])
        output = output.squeeze()
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.out(output)

        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, (d_model+1)//2*2)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, (d_model+1)//2*2, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:,0:d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

model = TransformerModel(1, 128, 8)
model.cuda()

trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(augmentProp=4, validationRatio=0.1, testRatio=0.1, flatten=False, denoise_n=8)

trainingX = torch.from_numpy(trainingX).cuda()
trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

validationX = torch.from_numpy(validationX).cuda()
validationX = validationX.transpose(0, 1)
testX = torch.from_numpy(testX).cuda()

trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

BATCH_SIZE = 250
MAX_ITER = 250

trainLoader = DataLoader(trainingDataset, BATCH_SIZE, True)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), weight_decay=0.1)

src_mask = model.generate_square_subsequent_mask(100).cuda()
for epoch in range(MAX_ITER):  # loop over the dataset multiple times
    running_loss = 0.0
    model.train()

    for i, data in enumerate(trainLoader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        inputs = inputs.transpose(0, 1)
        logits = model(inputs, src_mask)
        loss = criterion(logits, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        model.eval()
        
        logits = model(validationX, src_mask)
        _, valPredicts = torch.max(logits, axis=1)
        valPredicts = valPredicts.cpu().numpy()
        print(classification_report(valPredicts, validationLabels))
    

print('Finished Training')
