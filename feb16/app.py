import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

data = pd.read_csv('data.csv')

features = torch.tensor(data.drop('Price',axis=1).to_numpy()).float()
target = torch.tensor(data['Price'].to_numpy()).float().reshape(-1,1)

fm = features.mean()
fs = features.std()
tm = target.mean()
ts = target.std()



print(fm)