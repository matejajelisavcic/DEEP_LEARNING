import torch
import numpy

X = torch.tensor(4)
w = torch.tensor(3)
b = torch.tensor(10)
Y = torch.tensor(13)

Yhat = X*w+b

loss = (Yhat - Y)**2

print(loss)