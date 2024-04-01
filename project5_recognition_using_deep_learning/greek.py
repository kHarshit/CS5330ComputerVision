"""
@author: Harshit Kumar, Khushi Neema
@brief: Train and test a simple neural network on the Greek dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.MyNetwork import MyNetwork
from utils import train_network

# Load the trained MNIST model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Freeze the entire network's parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer to recognize 3 classes (alpha, beta, gamma)
model.fc2 = nn.Linear(50, 3)

class GreekTransform:
    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale(x)
        x = transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = transforms.functional.center_crop(x, (28, 28))
        return transforms.functional.invert(x)

training_set_path = './data/greek_train/'

greek_dataset = ImageFolder(root=training_set_path,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                GreekTransform(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))

greek_train_loader = DataLoader(greek_dataset, batch_size=5, shuffle=True)

optimizer = optim.Adam(model.fc2.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the network
train_losses, train_acc = train_network(model, greek_train_loader, optimizer, criterion)

# plot the training loss and accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_acc)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

