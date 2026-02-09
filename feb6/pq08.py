import torch

X = torch.tensor([
    [5.0]
], requires_grad=True)

f = X**3
f.backward()
print(X.grad)