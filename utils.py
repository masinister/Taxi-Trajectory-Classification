import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import datetime

long_mean, long_std, lat_mean, lat_std = (114.0471251204773, 0.09265583326259941, 22.555874215757328, 0.04737859070837168)

max_traj_len = 4416

def encode_data(long, lat, time, status):
    norm_long = (long - long_mean) / long_std
    norm_lat = (lat - lat_mean) / lat_std
    return [norm_long, norm_lat] + convert_time(time) + [status]

def sin_cos(n):
    theta = 2 * np.pi * n
    return (np.sin(theta), np.cos(theta))

def convert_time(time):
    d = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return list(sum(map(lambda t: sin_cos(t), [(d.month - 1)/ 12, (d.day - 1) / months[d.month - 1], d.weekday() / 7, d.hour / 24, d.minute / 60, d.second / 60]), ()))

def traj_to_tensor(traj, n_features, pad_len):
    result = np.zeros((pad_len, n_features))
    feature = np.stack(list(map(lambda d: encode_data(d[0], d[1], d[2], d[3]), traj)))
    result[:feature.shape[0]] = feature
    return torch.tensor(result, dtype=torch.float32, device = torch.device("cuda"))

"""
Sort trajectories by time, retain labels
"""
def aggr(data):
    traj_raw = data.values[:,1:]
    traj = np.array(sorted(traj_raw, key = lambda d:d[2]))
    label = data.iloc[0][0]
    return [traj, label]

"""
Split trajectory into smaller chunks
"""
def split_traj(traj, max_len):
    result = []
    subtraj = [traj[0]]
    for t in traj:
        if len(subtraj) < max_len:
            subtraj.append(t)
        else:
            result.append(subtraj)
            subtraj = [t]
    return result

"""
Parse CSV file
Input: file path
Returns: train_data, train_labels
"""
def parse_file(path, traj_len):
    data = pd.read_csv(path)

    plate_data = data.groupby('plate').apply(aggr)

    train_data = []
    train_labels = []

    for plate in plate_data:
        label = plate[1]
        traj = plate[0]
        for subtraj in split_traj(traj, max_len = traj_len):
            train_data.append(subtraj)
            train_labels.append(label)

    return train_data, train_labels

def train_test_split(dataset, test_split, batch_size):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_size = int(test_split * dataset_size)
    test_idx = np.random.choice(indices, size=test_size, replace=False)
    train_idx = list(set(indices) - set(test_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = DataLoader(dataset, batch_size = batch_size, sampler=train_sampler)
    testloader = DataLoader(dataset, batch_size = batch_size, sampler=test_sampler)

    return trainloader, testloader

"""
Only needs to be done once.
"""
def max_traj_length():
    maxlen = 0
    directory = 'data_5drivers'
    for entry in os.scandir(directory):
        if entry.path.endswith(".csv") and entry.is_file():
            data = pd.read_csv(entry.path)
            plate_data = data.groupby('plate').apply(aggr)
            for plate in plate_data:
                traj = plate[0]
                subtraj = [traj[0]]
                for t in traj:
                    if t[-1] == subtraj[-1][-1]:
                        subtraj.append(t)
                    else:
                        maxlen = max(maxlen, len(subtraj))
                        subtraj = [t]
    return maxlen

"""
Only needs to be done once.
"""
def get_coord_dist():
    longs = []
    lats = []
    directory = 'data_5drivers'
    for entry in os.scandir(directory):
        if entry.path.endswith(".csv") and entry.is_file():
            data = pd.read_csv(entry.path)
            longs += list(data['longitude'])
            lats += list(data['latitude'])
    longs = [x for x in longs if x >= 110 and x <= 115]
    lats = [x for x in lats if x >= 20 and x <= 25]
    return np.mean(longs), np.std(longs), np.mean(lats), np.std(lats)



if __name__ == '__main__':
    print("Utils test:")
    # print(max_traj_length())
    print(get_coord_dist())
