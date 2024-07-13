import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper paramiter
input_size = 28 * 28
hidden_size = 1000
n_classes = 10
n_epoch = 10
batch_size = 100
learning_rate = 0.001

# mnist
train_dataset = dataset.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dataset.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

examples = iter(train_loader)
samples, labels = examples.__next__()
# print(samples.shape,labels.shape)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')
    # plt.show()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        return self.l2(out)


model = NeuralNet(input_size, hidden_size, n_classes)

# loss and optimazer
criterion = nn.CrossEntropyLoss()
optimazer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# treaning model

n_total_step = len(train_loader)

for epoch in range(n_epoch):
    for i, (images, labels) in enumerate(train_loader):
        # 100 , 1 , 28 , 28
        # 100, 784
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backword
        optimazer.zero_grad()
        loss.backward()
        optimazer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {n_epoch}, step {i + 1}/ {n_total_step}, loss {loss.item():.4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        # val, index
        _,pred = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (pred == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'acc = {acc:.4f}')