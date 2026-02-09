import torch 

X = torch.tensor([
    [3.0]
], requires_grad=True)

Y = torch.tensor([
    [10.0]
], requires_grad=True)

f = X**2 + Y**2
f.backward()
print(X.grad, Y.grad)