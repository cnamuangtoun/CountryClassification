from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch
from tqdm import tqdm
from config import data_path, num_classes, batch_size, num_epochs, learning_rate
from sklearn.metrics import confusion_matrix
from models import *
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available()else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  
])

train_dataset = datasets.ImageFolder(
    root=data_path + 'train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(
    root=data_path + 'test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = baseline(num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)


train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


total_steps = len(train_loader)
highest_accuracy = 0
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader, total=total_steps,
                desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    total_loss = 0
    total_accuracy = 0
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_accuracy += (outputs.argmax(1) == labels).sum().item()
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'Loss': loss.item()})
    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(total_accuracy / len(train_dataset))

    model.eval()
    with torch.no_grad():

        total_loss = 0

        all_predicted = []
        all_labels = []
        correct = 0
        total = 0
        testbar = tqdm(test_loader, total=len(test_loader),
                       desc=f'Test', unit='batch')


        for images, labels in testbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += criterion(outputs, labels).item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            testbar.set_postfix({'Accuracy': f'{100 * correct / total:.2f}%'})

        test_losses.append(total_loss / len(test_loader))
        test_accuracies.append(correct / total)

        if correct / total > highest_accuracy:
            highest_accuracy = correct / total
            torch.save(model.state_dict(), 'best_model.pth')

        confusion = confusion_matrix(all_labels, all_predicted)
        print('Confusion Matrix:')
        print(confusion)
    scheduler.step()

plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.clf()

plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
