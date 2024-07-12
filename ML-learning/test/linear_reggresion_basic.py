# 1. Desing model (input, output size, forward pass)
# 2. Construct loss and optimizer
# 3. Training loop
#     - forward pass: compute prediction
#     - backward pass: gradients
#     - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Prepare data

x_numpy, y_numpy = make_regression(
    n_samples=100,
    n_features=1,
    noise=20,
    random_state=4
)

X = torch.from_numpy(x_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32))

Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape

# model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

# loss and optimazer
criterion = nn.MSELoss()

from torch.optim import SGD

learning_rate = 0.01
optimizer = SGD(model.parameters(), lr=learning_rate)

# treaning loop

n_epoch = 100

for epoch in range(n_epoch):
    # forward
    y_predicted = model(X)
    loss = criterion(y_predicted, Y)

    # backwalk
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch + 1}: loss = {loss.item():.8f}')

# plot
predicted = model(X).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()
