import os
from time import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

from sklearn import datasets
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.quantization import QuantStub, DeQuantStub

def time_model(model, test_X):
    model.eval()
    i = 0
    time_spent = []
    while i < 500:
        start_time = time()
        with torch.no_grad():
            _ = model(test_X)
        if i != 0:
            time_spent.append(time() - start_time)
        i += 1
    print('\tAvg execution time per forward(ms): {:.5f}'.format(np.mean(time_spent)))

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print('\tSize (MB):', size / 1e6)
    os.remove('temp.p')
    return size


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 3)
        self.softmax = nn.Softmax(dim=1)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, X):
        X = self.quant(X)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.dequant(X)
        return X


# load IRIS dataset
dataset = datasets.load_wine()

train_X, test_X, train_y, test_y = train_test_split(dataset.data,
                                                    dataset.target, test_size=0.8)

# wrap up with Variable in pytorch
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())

net = Net()
criterion = nn.CrossEntropyLoss()  # cross entropy loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(500):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('number of epoch', epoch, 'loss', loss.data)

predict_out = net(test_X)
predict_loss = criterion(predict_out, test_y.data)
_, predict_y = torch.max(predict_out, 1)
print('acc:', accuracy_score(test_y.data, predict_y.data))
print('loss: ', predict_loss.data.item())
print_size_of_model(net)
time_model(net, test_X)

###
quantized_model = torch.quantization.quantize_dynamic(net.eval(), {torch.nn.Linear}, dtype=torch.qint8)
# print(quantized_model)

predict_out = quantized_model(test_X)
predict_loss = criterion(predict_out, test_y.data)
_, predict_y = torch.max(predict_out, 1)
print('acc:', accuracy_score(test_y.data, predict_y.data))
print('loss: ', predict_loss.data.item())
print_size_of_model(quantized_model)
time_model(quantized_model, test_X)


###
quantized_model = net
quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
quantized_model = torch.quantization.prepare(quantized_model)
optimizer.zero_grad()
out = quantized_model(train_X)
loss = criterion(out, train_y)
loss.backward()
optimizer.step()
quantized_model = torch.quantization.convert(quantized_model)

predict_out = quantized_model(test_X)
predict_loss = criterion(predict_out, test_y.data)
_, predict_y = torch.max(predict_out, 1)
print('acc:', accuracy_score(test_y.data, predict_y.data))
print('loss: ', predict_loss.data.item())
print_size_of_model(quantized_model)
time_model(quantized_model, test_X)

###
quantized_model = Net()
quantized_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
criterion = nn.CrossEntropyLoss()  # cross entropy loss
optimizer = torch.optim.Adam(quantized_model.parameters(), lr=0.0001)
torch.quantization.prepare(quantized_model, inplace=True)
for epoch in range(500):
    optimizer.zero_grad()
    out = quantized_model(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('number of epoch', epoch, 'loss', loss.data)

quantized_model = torch.quantization.convert(quantized_model.eval(), inplace=False)
predict_out = quantized_model(test_X)
predict_loss = criterion(predict_out, test_y.data)
_, predict_y = torch.max(predict_out, 1)
print('acc:', accuracy_score(test_y.data, predict_y.data))
print('loss: ', predict_loss.data.item())
print_size_of_model(quantized_model)
time_model(quantized_model, test_X)

# result fbgemm is always worse than dynamic...should no be the case -> official bug on local machine