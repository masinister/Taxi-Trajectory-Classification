import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os

def train(model, trainloader, testloader, device, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr = 1e-4, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        bar = tqdm(enumerate(trainloader, 0), total = len(trainloader))
        for i, (inp,lab) in bar:
            inputs, labels = inp.to(device), lab.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            bar.set_description('Epoch %d, loss: %.3f, acc: %.3f'%(epoch + 1, running_loss, correct/total))

        correct, total = test(model, testloader, device)
        print("Validation: %d/%d=%.2f Accuracy."%(correct, total, correct/total))

    return model

def test(model, dataloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            trajectories, labels = data
            outputs = model(trajectories)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total
