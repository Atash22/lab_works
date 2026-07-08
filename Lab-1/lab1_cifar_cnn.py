"""
Lab-1: CNN Networks for CIFAR-10 and CIFAR-100
Two different networks with convolution, padding, stride, pooling, and dropout.

Student note:
- Network 1 -> CIFAR-10 (10 classes)
- Network 2 -> CIFAR-100 (100 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# -----------------------------
# 1. Data preparation
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 dataset
trainset_10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset_10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader_10 = torch.utils.data.DataLoader(trainset_10, batch_size=64, shuffle=True)
testloader_10 = torch.utils.data.DataLoader(testset_10, batch_size=64, shuffle=False)

# CIFAR-100 dataset
trainset_100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset_100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader_100 = torch.utils.data.DataLoader(trainset_100, batch_size=64, shuffle=True)
testloader_100 = torch.utils.data.DataLoader(testset_100, batch_size=64, shuffle=False)


# -----------------------------
# 2. Network 1 -> CIFAR-10
# -----------------------------
class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layer
        self.dropout = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)   # 10 classes for CIFAR-10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -----------------------------
# 3. Network 2 -> CIFAR-100
# -----------------------------
class CIFAR100_Net(nn.Module):
    def __init__(self):
        super(CIFAR100_Net, self).__init__()

        # Convolution layers (deeper network for more classes)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layer
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 100)   # 100 classes for CIFAR-100

    def forward(self, x):
        x = F.relu(self.conv1(x))              # 32x32
        x = self.pool(F.relu(self.conv2(x)))    # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv3(x)))    # 16x16 -> 8x8
        x = self.pool(x)                        # 8x8 -> 4x4
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -----------------------------
# 4. Simple training function
# -----------------------------
def train_model(model, trainloader, epochs=2, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f}")

    return model


# -----------------------------
# 5. Simple test/accuracy function
# -----------------------------
def test_model(model, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# -----------------------------
# 6. Run everything
# -----------------------------
if __name__ == "__main__":
    print("Training Network 1 on CIFAR-10...")
    model10 = CIFAR10_Net()
    model10 = train_model(model10, trainloader_10, epochs=2)
    test_model(model10, testloader_10)

    print("\nTraining Network 2 on CIFAR-100...")
    model100 = CIFAR100_Net()
    model100 = train_model(model100, trainloader_100, epochs=2)
    test_model(model100, testloader_100)
