import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()

first_data = dataset[0]

features, labels = first_data
# print(features, labels)

dataloaer = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

# datatiter = iter(dataloaer)
# data = datatiter.__next__()
# features, labels = data
# print(features, labels)


n_epoch = 2
total_samples = len(dataset)
n_iteration = math.ceil(total_samples / 4)

print(total_samples, n_iteration)

for epoch in range(n_epoch):
    for i, (inputs, labels) in enumerate(dataloaer):
        if (i + 1) % 5 == 0:
            print(f'epoch {epoch + 1}/{n_epoch}, step {i + 1}/{n_iteration}, inputs {inputs.shape}')
