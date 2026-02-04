import torch

X = torch.tensor([
    [2.0],
    [5.0]
])

Y = torch.tensor([
    [5.0],
    [1.0]
])

w = torch.tensor([
    [3.0]
])

b = torch.tensor([
    [1.0]
])

Yhat = X@w + b

r = Yhat - Y 
SSE = r.T@r # Squaring
loss = SSE / X.shape[0]  # Mean X.shape[0] = number of samples ~ 2

print(loss.item())