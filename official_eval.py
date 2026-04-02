import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA
transform = transforms.Compose([
    transforms.ToTensor(),
])

test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(test, batch_size=64)

# MODEL (OFFICIAL PRETRAINED)
model = resnet18(weights="DEFAULT")

# MODIFY FINAL LAYER
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

# PARAM COUNT
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

official_params = count_params(model)

model.eval()
correct, total = 0, 0

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

official_acc = 100 * correct / total

print("Official Accuracy:", official_acc)
print("Parameters:", official_params)

# READ CUSTOM RESULTS
with open("custom_results.txt", "r") as f:
    custom_data = f.read()

# SAVE FINAL COMPARISON
with open("final_results.txt", "w") as f:
    f.write("=== CUSTOM MODEL ===\n")
    f.write(custom_data)
    f.write("\n=== OFFICIAL MODEL ===\n")
    f.write(f"Accuracy: {official_acc:.2f}\n")
    f.write(f"Parameters: {official_params}\n")

print("Comparison saved in final_results.txt")