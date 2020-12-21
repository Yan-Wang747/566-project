import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import math
from preprocessing import loadData
from sklearn.metrics import classification_report

import random
import shared
import numpy as np

import matplotlib.pyplot as plt

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

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    return mask

mode = shared.SPLIT_MODE_CLASSIC

if mode == shared.SPLIT_MODE_CLASSIC:
    trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(normalize=False)
    trainingX = torch.from_numpy(trainingX).cuda()
    trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

    validationX = torch.from_numpy(validationX).cuda()
    validationX = validationX.transpose(0, 1)

    testX = torch.from_numpy(testX).cuda()
    testX = testX.transpose(0, 1)

    trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

    reportName = "trans_report_rand_no_norm.txt"
    
elif mode == shared.SPLIT_MODE_BY_SUBJECT:
    reportName = "trans_report_ind.txt"

report = open(reportName, "w")
report.close()

runs = 1
BATCH_SIZE = 250
MAX_ITER = 500
criterion = nn.CrossEntropyLoss()
src_mask = generate_square_subsequent_mask(100).cuda()
for r in range(runs):
    print("r = " + str(r))

    if mode == shared.SPLIT_MODE_BY_SUBJECT:
        subject = random.choice(shared.SUBJECTS)
        trainingX, trainingLabels, validationX, validationLabels, testX, testLabels = loadData(subjects=[subject])

        trainingX = torch.from_numpy(trainingX).cuda()
        trainingLabels = torch.from_numpy(trainingLabels).long().cuda()

        validationX = torch.from_numpy(validationX).cuda()
        validationX = validationX.transpose(0, 1)
        testX = torch.from_numpy(testX).cuda()
        testX = testX.transpose(0, 1)

        trainingDataset = torch.utils.data.TensorDataset(trainingX, trainingLabels)

    trainLoader = DataLoader(trainingDataset, BATCH_SIZE, True)

    model = TransformerModel(3, 128, 8)
    model.cuda()
    optimizer = optim.AdamW(model.parameters(), weight_decay=0.2)

    losses = []
    accs = []
    best_val_acc = -1
    best_epoch = None

    for epoch in range(MAX_ITER):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        print("epoch: " + str(epoch))
        for i, data in enumerate(trainLoader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.transpose(0, 1)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            logits = model(inputs, src_mask)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        losses.append(running_loss)

        with torch.no_grad():
            model.eval()

            logits = model(validationX, src_mask)
            _, predicts = torch.max(logits, axis=1)
            predicts = predicts.cpu().numpy()
            acc = np.mean(predicts == validationLabels)
            accs.append(acc)
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(model.state_dict(), './best_transformer_model.pth')
                best_epoch = epoch
        
    print(best_epoch)
    print(best_val_acc)

    bestModel = TransformerModel(3, 128, 8)
    bestModel.load_state_dict(torch.load('best_transformer_model.pth'))
    bestModel.to('cuda')
    bestModel.eval()

    with torch.no_grad():
        logits = bestModel(testX, src_mask)
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