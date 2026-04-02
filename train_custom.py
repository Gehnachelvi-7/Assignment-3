import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from custom_resnet import SmallResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=64)

model = SmallResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# TRAIN (FAST)
for epoch in range(3):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")

# EVALUATE
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

acc = 100 * correct / total
print("Custom Accuracy:", acc)

torch.save(model.state_dict(), "custom_model.pth")

with open("custom_results.txt", "w") as f:
    f.write(str(acc))