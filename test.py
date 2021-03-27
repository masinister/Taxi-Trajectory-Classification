import pandas as pd
import numpy as np
import pickle
import torch
from evaluation import *

model = torch.load('testmodel.pth')
test_data = pickle.load(open('test.pkl','rb'))

for te in test_data:
    dt = process_data(te)
    re = run(dt, model)
