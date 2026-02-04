import torch

X = torch.tensor([
    [2.0,3.0]
])

Y = torch.tensor([
    [30.0]
])

w = torch.tensor([
    [4.0],
    [5.0]
])

b = torch.tensor([
    [1.0]
])

Yhat = X@w + b
r = Yhat - Y 
SSE = r.T@r # Squaring ~ Sum of Squared Errors
loss = SSE / 1  # Mean X.shape[0] = number of samples ~ 1
print(loss.item())