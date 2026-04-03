import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from torchvision import datasets, transforms
from custom_resnet import SmallResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DATA - with augmentation as in the paper
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True,  num_workers=2)
test_loader  = torch.utils.data.DataLoader(test,  batch_size=128, shuffle=False, num_workers=2)

# MODEL
model = SmallResNet().to(device)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

custom_params = count_params(model)
print(f"Custom model parameters: {custom_params:,}")

# LOSS + OPTIMIZER - SGD with momentum + weight decay (paper-faithful)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)

num_epochs = 10
train_losses, train_accs, test_accs = [], [], []

total_start = time.time()

for epoch in range(num_epochs):
    start = time.time()
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    train_acc = 100 * correct / total
    avg_loss  = total_loss / len(train_loader)

    # EVAL
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total   += y.size(0)

    test_acc = 100 * correct / total
    epoch_time = time.time() - start

    train_losses.append(avg_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.3f} | "
          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
          f"Time: {epoch_time:.1f}s")

    scheduler.step()

total_time = time.time() - total_start
custom_acc = test_accs[-1]

print(f"\nFinal Test Accuracy: {custom_acc:.2f}%")
print(f"Total Training Time: {total_time:.1f}s")
print(f"Parameters: {custom_params:,}")

# SAVE MODEL
torch.save(model.state_dict(), "custom_model.pth")

# SAVE RESULTS
with open("custom_results.txt", "w") as f:
    f.write(f"Accuracy: {custom_acc:.2f}\n")
    f.write(f"Training Time: {total_time:.2f}\n")
    f.write(f"Parameters: {custom_params}\n")

# SAVE CURVE DATA for plotting
with open("custom_curves.json", "w") as f:
    json.dump({
        "train_losses": train_losses,
        "train_accs":   train_accs,
        "test_accs":    test_accs
    }, f)

print("Saved: custom_model.pth, custom_results.txt, custom_curves.json")
