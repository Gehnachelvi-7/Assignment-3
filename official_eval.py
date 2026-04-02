import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision.models import resnet18
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA
transform = transforms.Compose([
    transforms.ToTensor(),
])

train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=64)

model = resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

official_params = count_params(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

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

    # EVALUATE AFTER EPOCH
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

# FINAL METRICS
official_acc = epoch_acc

print("\nFinal Official Accuracy:", official_acc)
print("Parameters:", official_params)

# READ CUSTOM RESULTS
with open("custom_results.txt", "r") as f:
    custom_data = f.read()

with open("final_results.txt", "w") as f:
    f.write("=== CUSTOM MODEL ===\n")
    f.write(custom_data)
    f.write("\n=== OFFICIAL MODEL ===\n")
    f.write(f"Accuracy: {official_acc:.2f}\n")
    f.write(f"Parameters: {official_params}\n")

print("\nComparison saved in final_results.txt")