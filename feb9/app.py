import torch

X = torch.tensor([
    [1.0],
    [5.0],
    [9.0]
])

Y = torch.tensor([
    [5.0],
    [8.0],
    [2.0]
])

w = torch.tensor([
    [0.0]
], requires_grad=True)

b = torch.tensor([
    [0.0]
], requires_grad=True)

lr = .01
for i in range(50000):
    Yhat = X@w+b
    r = Yhat-Y
    SSE = r.T@r
    loss = SSE/3

    loss.backward()
    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
    print(loss.item(), w, b)

    w.grad.zero_()
    b.grad.zero_()

print(7.0*w+b)
# -0.3723 6.8561