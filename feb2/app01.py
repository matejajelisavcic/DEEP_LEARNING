import torch

x = torch.tensor(7.0, requires_grad=True)
f = (x**2+1)/(x+5)
f.backward()
print(x.grad)  