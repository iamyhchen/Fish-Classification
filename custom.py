import os
import sys
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# hyperparameters setting
batch_size = 32
epochs = 50
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# load dataset
train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
test_dataset = datasets.ImageFolder("dataset/test", transform=transform)
val_dataset = datasets.ImageFolder("dataset/eval", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# output directories setting
model_dir = f"exp/checkpoint/nn"
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

result_dir = f"exp/result/nn"
if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True)

# model, loss function and optimizer setting
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)  # 112x112
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  # 56x56
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)  # 28x28
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(28 * 28 * 256, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.act5(x)
        x = self.conv6(x)
        x = self.act6(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# early stopping setting
best_val_loss = 100
patience = 7
no_improve_epochs = 0

# loss and accuracy lists for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# start training
print(f"start training: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
for epoch in range(epochs):
    # train
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total

    # evaluate
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total
    
    print(
        f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), f"{model_dir}/best_model.pth")
    else:
        no_improve_epochs += 1
    if no_improve_epochs >= patience:
        torch.save(model.state_dict(), f"{model_dir}/last_model.pth")
        print("Early stopping triggered.")
        break
print(f"finish training: { datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

# evaluate the best model
model.load_state_dict(torch.load(f"{model_dir}/best_model.pth"))
model.to(device)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print(f"Train Accuracy: {accuracy:.4f}")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print(f"val Accuracy: {accuracy:.4f}")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# make loss and accuracy plots
epochs = range(len(train_losses))  
plt.figure(figsize=(5, 4))
plt.plot(epochs, train_losses, color="blue", label="Train Loss")
plt.plot(epochs, val_losses, color="orange", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VGG16 Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{result_dir}/loss.png")
# plt.show()

plt.figure(figsize=(5, 4))
plt.plot(epochs, train_accuracies, color="blue", label="Train Accuracy")
plt.plot(epochs, val_accuracies, color="orange", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("VGG16 Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"{result_dir}/accuracy.png")
# plt.show()

# make confusion matrix
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix, display_labels=train_dataset.classes
)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
plt.tight_layout()
plt.xlabel("predicted")
plt.ylabel("ground truth")
plt.savefig(f"{result_dir}/confusion_matrix.png")
# plt.show()


