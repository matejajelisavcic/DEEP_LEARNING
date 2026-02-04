import torch

X = torch.tensor([
    [1.0],
    [5.0],
    [8.0]
])

Y = torch.tensor([
    [3.0],
    [6.0],
    [1.0]
])

w  = torch.tensor([
    [0.0]
], requires_grad=True)

b = torch.tensor([
    [0.0]
], requires_grad=True)

lr = .01

for i in range(1000):
    Yhat = X@w+b
    r = Yhat-Y
    SSE = r.T@r
    loss = SSE/3

    loss.backward()

    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
    print(w, b)

    w.grad.zero_()
    b.grad.zero_()
    print(w, b) 


#-0.2265 4.851