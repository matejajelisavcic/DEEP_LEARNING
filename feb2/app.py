import torch

x = torch.tensor(3.0,requires_grad=True)
f = x**2
f.backward()
print(x.grad)  # Should print tensor(6.0) since df/dx = 2x and x=3.0
x.grad.zero_()  # Reset gradients to zero

f = x**2
f.backward()
print(x.grad)  # Should print tensor(12.0) since gradients accumulate: