import os
from time import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

from sklearn import datasets
from sklearn.svm import SVC
import torchvision.datasets as datasets
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.quantization import QuantStub, DeQuantStub

def time_test(model, test_X, test_y):
    model.to('cpu')
    start_time = time()
    predict_out = model(test_X)
    predict_loss = criterion(predict_out, test_y.data)
    _, predict_y = torch.max(predict_out, 1)
    end_time = time() - start_time
    print('\nacc:', accuracy_score(test_y.data, predict_y.data))
    print('loss: ', predict_loss.data.item())
    #print('time (ms): {:.5f}\n'.format(np.mean(end_time)))

def time_model(model, test_X):
    model.to('cpu')
    model.eval()
    i = 0
    time_spent = []
    while i < 10:
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
        self.fc1 = nn.Linear(109, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 2)
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


# load dataset
dataset = pd.read_csv("data/census_income_scikit_onehot_dataset.csv")
labels = dataset.income_level
dataset.drop(['income_level'], axis=1)

train_X, test_X, train_y, test_y = train_test_split(dataset,
                                                    labels, test_size=0.8)

# wrap up with Variable in pytorch
train_X = Variable(torch.Tensor(train_X.to_numpy()).float())
test_X = Variable(torch.Tensor(test_X.to_numpy()).float())
train_y = Variable(torch.Tensor(train_y.to_numpy()).long())
test_y = Variable(torch.Tensor(test_y.to_numpy()).long())


net = Net()
criterion = nn.CrossEntropyLoss()  # cross entropy loss
optimizer = torch.optim.SGD(net.parameters(), lr=0.0000000001)

net.cuda()
for epoch in range(500):
    optimizer.zero_grad()
    train_X_cuda = train_X.cuda()
    train_y_cuda = train_y.cuda()
    out = net(train_X_cuda)
    loss = criterion(out, train_y_cuda)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('number of epoch', epoch, 'loss', loss.data)

time_test(net, test_X, test_y)
print_size_of_model(net)
time_model(net, test_X)

net.to('cpu')
###
quantized_model = torch.quantization.quantize_dynamic(net.eval(), {torch.nn.Linear}, dtype=torch.qint8)
# print(quantized_model)

time_test(quantized_model, test_X, test_y)
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

time_test(quantized_model, test_X, test_y)
print_size_of_model(quantized_model)
time_model(quantized_model, test_X)

###
quantized_model = Net()
criterion = nn.CrossEntropyLoss()  # cross entropy loss
optimizer = torch.optim.SGD(quantized_model.parameters(), lr=0.0000000001)

quantized_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare(quantized_model, inplace=True)

quantized_model.cuda()
for epoch in range(500):
    optimizer.zero_grad()
    train_X_cuda = train_X.cuda()
    train_y_cuda = train_y.cuda()
    out = quantized_model(train_X_cuda)
    loss = criterion(out, train_y_cuda)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('number of epoch', epoch, 'loss', loss.data)

quantized_model.to('cpu')
quantized_model = torch.quantization.convert(quantized_model.eval(), inplace=False)
time_test(quantized_model, test_X, test_y)
print_size_of_model(quantized_model)
time_model(quantized_model, test_X)

# result fbgemm is always worse than dynamic...should not be the case -> official bug on local machine