import torch

x = torch.rand(3, requires_grad=True)

print(x)

y = x + 2
print(y)

z = y * y * 2
# z = z.mean()
print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)

z.backward(v)
print(x.grad)

x = torch.rand(3, requires_grad=True)
print(x)

# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():


weight = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weight * 3).sum()
    model_output.backward()
    print(weight.grad)
    weight.grad.zero_()  # this is important, otherwise the gradients will accumulate

from torch.optim import SGD

optimizer = SGD([weight], lr=0.01)
optimizer.step()
optimizer.zero_grad()

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss

y_hat = w * x
loss = (y_hat - y) ** 2
print(loss)

# backward pass

loss.backward()
print(w.grad)
