import torch
from torchvision.models import resnet18
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(test, batch_size=64)

model = resnet18(num_classes=10).to(device)

model.eval()
correct, total = 0, 0

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

acc = 100 * correct / total
print("Official Accuracy:", acc)

# READ custom
with open("custom_results.txt", "r") as f:
    custom_acc = float(f.read())

with open("final_results.txt", "w") as f:
    f.write(f"Custom: {custom_acc}\nOfficial: {acc}")