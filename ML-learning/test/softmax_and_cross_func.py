import torch
import torch.nn as nn
import numpy as np
from numpy import ndarray


def softmax(x: ndarray) -> ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 0.01])
print('softmax numpy:', softmax(x))

x = torch.tensor([2.0, 1.0, 0.01])
output = torch.softmax(x, dim=0)
print('softmax torch:', output)


def cross_entropy(actual: ndarray, predict: ndarray) -> ndarray:
    loss = - np.sum(actual * np.log(predict))
    return loss  # / float(predict.shape[0])


# y musi mieć jedno kodowanie
# klasa 0: [1 0 0]
# klasa 1: [0 1 0]
# klasa 2: [0 0 1]

y = np.array([1, 0, 0])
y_good_pred = np.array([0.7, 0.2, 0.1])
y_bad_pred = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(y, y_good_pred)
l2 = cross_entropy(y, y_bad_pred)

print(f'loss 1: {l1:.4f}')
print(f'loss 2: {l2:.4f}')

loss = nn.CrossEntropyLoss()

# 3 samples

y = torch.tensor([2, 0, 1])

# n_sumples * n_classes
y_good_pred = torch.tensor([
    [0.01, 1.0, 2.1],
    [2.0, 1.0, 0.1],
    [0.9, 3.0, 0.1]
])
y_bad_pred = torch.tensor([
    [0.5, 3.0, 0.2],
    [0.5, 3.0, 0.2],
    [0.5, 0.01, 0.2]
])

l1 = loss(y_good_pred, y)
l2 = loss(y_bad_pred, y)

print(f'loss 1: {l1:.4f}')
print(f'loss 2: {l2:.4f}')

_, pred1 = torch.max(y_good_pred, 1)
_, pred2 = torch.max(y_bad_pred, 1)

print(pred1)
print(pred2)


class NeuralNetMultiClass(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(NeuralNetMultiClass, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU
        self.linear2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # brak fun. softmax na końcu
        return out


model = NeuralNetMultiClass(input_size=28 * 28, hidden_size=5, n_classes=3)
criterion = nn.CrossEntropyLoss()  # przymuje funk. softmax !!!


class NeuralNetBinClass(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(NeuralNetBinClass, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # funk. sigmoid na końcu
        return torch.sigmoid(out)


model = NeuralNetBinClass(input_size=28 * 28, hidden_size=5)
criterion = nn.BCELoss()
