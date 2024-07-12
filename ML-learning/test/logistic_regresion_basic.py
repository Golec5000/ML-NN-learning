# 1. Desing model (input, output size, forward pass)
# 2. Construct loss and optimizer
# 3. Training loop
#     - forward pass: compute prediction
#     - backward pass: gradients
#     - update weights

import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

n_samples, n_features = x.shape
# print(n_samples,n_features)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1234)

# scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# model
# f = wx + b, sigmoid at the end

class Model(nn.Module):
    def __init__(self, n_input_featutes):
        super().__init__()
        self.lin = nn.Linear(n_input_featutes, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.lin(x))
        return y_predicted


model = Model(n_features)

# loss

criterion = nn.BCELoss()

from torch.optim import SGD

optimizer = SGD(model.parameters(), lr=0.01)

# trainnng loop

n_epoch = 100

for epoch in range(n_epoch):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch + 1}: loss = {loss.item():.8f}')

with torch.no_grad():
    y_pred = model(x_test)
    y_pred_class = y_pred.round()
    acc = y_pred_class.eq(y_test).sum() / float(y_test.shape[0])
    print(acc.item())
