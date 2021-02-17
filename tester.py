from torch.nn.utils import prune
import torch
from model import DeepFMs
from utils.util import get_model, load_model_dic, get_logger
import time
import numpy as np
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def computeTime(model):
    inputs = torch.randn(256, 1, 28, 28)

    model.eval()

    i = 0
    time_spent = []
    while i < 100:
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    print('Avg execution time (ms): {:.3f}'.format(np.mean(time_spent)))

logger = get_logger()
model = LeNet()#DeepFMs.DeepFMs(field_size=23, feature_sizes=[1], logger=logger)

no_non_sparse = 0
for name, param in model.named_parameters():
    no_non_sparse += (param != 0).sum().item()
print(no_non_sparse)
computeTime(model)

prune.ln_structured(model.fc1, name="weight", amount=0.5, n=2, dim=0)
prune.remove(model.fc1, 'weight')

no_non_sparse = 0
for name, param in model.named_parameters():
    no_non_sparse += (param != 0).sum().item()
print(no_non_sparse)
state_dict = model.state_dict()
print(state_dict.keys())

model = LeNet()#DeepFMs.DeepFMs(field_size=23, feature_sizes=[1], logger=logger)
model.load_state_dict(state_dict)
no_non_sparse = 0
for name, param in model.named_parameters():
    no_non_sparse += (param != 0).sum().item()
print(no_non_sparse)
computeTime(model)
