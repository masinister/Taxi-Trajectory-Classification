import pandas as pd
import numpy as np
import pickle
import torch
from utils import split_traj, traj_to_tensor, aggr
from model import Trajectory_NN, Trajectory_RNN

#def run and predict
def process_data(traj, len = 32):
    return [traj_to_tensor(subtraj, n_features = 15, pad_len = len) for subtraj in split_traj(traj, max_len = len)]

def run(data, model):
    # Initialize model
    # net = Trajectory_NN(input_size = 15, traj_len = 100).to(torch.device("cuda"))
    net = Trajectory_RNN(input_size = 15).to(torch.device("cuda"))
    net.load_state_dict(model)
    net.eval()

    pred = np.zeros(5)
    for s in data:
        with torch.no_grad():
            i = net.predict(s.unsqueeze(0))
        pred[i.item()] += 1
    return np.argmax(pred)
