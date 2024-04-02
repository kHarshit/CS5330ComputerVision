import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from models.ConvAutoencoder import ConvAutoencoder
from utils import train_autoencoder

# convert to tensor
transform = transforms.ToTensor()

batch_size = 64
# load the training and test datasets
train_set = MNIST(root='./data', train=True, download=True, transform=transform)
test_set = MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# instatntiate the model
model = ConvAutoencoder()
print(model)
if 0:
    #To see the diagram of the model architecture
    dummy_input = torch.randn(1, 1, 28, 28)
    writer = SummaryWriter('runs/autoencoder_model')
    writer.add_graph(model, dummy_input)
    writer.close()

# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_acc = train_autoencoder(model, train_loader, optimizer, criterion, device, n_epochs)

# save model
torch.save(model.state_dict(), 'conv_autoencoder.pth')

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)

# get sample outputs
output = model(images)
# prep images for display
images = images.numpy()

# output is resized into a batch of iages
output = output.view(batch_size, 1, 28, 28)
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

titles = ['Input Images', 'Reconstructions']
for images, row, title in zip([images, output], axes, titles):
    row[0].set_title(title)  # Set title for the row
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()
