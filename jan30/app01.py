import torch
import pandas as pd
import numpy as np

df = pd.read_csv('data01.csv')

X = torch.tensor((df.drop("Y", axis=1)).to_numpy()).float()

Y = torch.tensor(df["Y"].to_numpy()).float().reshape(-1, 1)

w = torch.tensor([
    [1.4],
    [0.4],
    [-2.3],
    [-2.3]
])

b = torch.tensor([
    [-2.4]
])

Yhat = X@w + b
r = Yhat - Y
SSE = r.T@r
loss = SSE / X.shape[0]

print(loss.item())