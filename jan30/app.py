import torch
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

X = torch.tensor(df.drop("Y", axis=1).to_numpy()).float()

Y = torch.tensor(df["Y"].to_numpy()).float().reshape(-1, 1)

w = torch.tensor([
    [-0.5],
    [-0.7]
])

b = torch.tensor([
    [1.5]
])

Yhat = X@w + b
r = Yhat - Y
SSE = r.T@r
loss = SSE / 2
print("Loss:", loss.item())
