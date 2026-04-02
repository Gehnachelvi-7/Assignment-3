import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from custom_resnet import SmallResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA
transform = transforms.Compose([
    transforms.ToTensor(),
])

train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=64)

# MODEL
model = SmallResNet().to(device)

# PARAM COUNT
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

custom_params = count_params(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# TRAIN + TIME
import time
start = time.time()

for epoch in range(3):  # FAST
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")

end = time.time()
train_time = end - start

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

custom_acc = 100 * correct / total

print("Custom Accuracy:", custom_acc)
print("Training Time:", train_time)
print("Parameters:", custom_params)

# SAVE EVERYTHING
torch.save(model.state_dict(), "custom_model.pth")

with open("custom_results.txt", "w") as f:
    f.write(f"Accuracy: {custom_acc:.2f}\n")
    f.write(f"Training Time: {train_time:.2f}\n")
    f.write(f"Parameters: {custom_params}\n")