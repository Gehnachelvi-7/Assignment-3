import torch
import torch.nn as nn
import torch.optim as optim
import time
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

# LOSS + OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# TRAINING
total_start = time.time()

num_epochs = 5

for epoch in range(num_epochs):
    start_time = time.time()

    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # EVALUATION AFTER EACH EPOCH
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

    epoch_acc = 100 * correct / total
    epoch_time = time.time() - start_time

    print(f"Epoch {epoch+1}: Loss={total_loss:.2f}, Accuracy={epoch_acc:.2f}%, Time={epoch_time:.2f}s")

# TOTAL TRAINING TIME
total_time = time.time() - total_start

# FINAL ACCURACY (already computed in last epoch)
custom_acc = epoch_acc

print("\nFinal Custom Accuracy:", custom_acc)
print("Total Training Time:", total_time)
print("Parameters:", custom_params)

# SAVE MODEL
torch.save(model.state_dict(), "custom_model.pth")

# SAVE RESULTS
with open("custom_results.txt", "w") as f:
    f.write(f"Accuracy: {custom_acc:.2f}\n")
    f.write(f"Training Time: {total_time:.2f}\n")
    f.write(f"Parameters: {custom_params}\n")