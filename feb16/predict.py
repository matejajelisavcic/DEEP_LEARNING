import torch
import torch.nn as nn

model_data = torch.load('model.pth')
fm = model_data['fm']
fs = model_data['fs']
tm = model_data['tm']
ts = model_data['ts']
parameters = model_data['parameters']

print(parameters)

feature = torch.tensor([
    [1500.0]
])

X = (feature - fm) / fs

model = nn.Linear(1,1)
model.load_state_dict(parameters)

prediction = model(X)
price = prediction * ts + tm
print(price)