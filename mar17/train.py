import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from export import export_model

data = pd.read_csv('data.csv')

X = torch.tensor(data.drop('y', axis=1).values).float()
Y = torch.tensor(data['y'].values).float().reshape(-1,1)

model = nn.Sequential(
    nn.Linear(2,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,1)
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()

export_model(model, 'model.json')