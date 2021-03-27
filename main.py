import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import os

from model import Trajectory_RNN, Trajectory_NN
from dataset import TrajectoryDataset
from learning import train, test
from utils import train_test_split

device = torch.device("cuda")
traj_len = 32
n_features = 15
test_split = 0.2
batch_size = 128
num_epochs = 10

model = Trajectory_RNN(input_size = n_features).to(device)
# model = Trajectory_NN(input_size = n_features, traj_len = traj_len).to(device)

dataset = TrajectoryDataset(dir = 'data_5drivers', n_features = n_features, traj_len = traj_len)

trainloader, testloader = train_test_split(dataset, test_split, batch_size)

print(len(dataset), len(trainloader), len(testloader), batch_size)

model = train(model, trainloader, testloader, device, num_epochs)

torch.save(model.state_dict(),"testmodel.pth")
