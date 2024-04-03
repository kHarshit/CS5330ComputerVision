"""
@author: Khushi Neema, Harshit Kumar
@brief: Visualize the filters of the first convolutional layer of a convolution layers of pre-trained resnet18.
"""

import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.models import resnet18
import cv2
import numpy as np



transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image or numpy.ndarray to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the image data
])
train_set = MNIST(root='./data', train=True, download=True, transform=transform)

model = resnet18(pretrained=True)
#model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#print(model)
    
conv_layers = [
    model.conv1,
    model.layer1[0].conv1,
    model.layer2[0].conv1,
    model.layer3[0].conv1,
    model.layer4[0].conv1
]

fig, axs = plt.subplots(len(conv_layers), 5, figsize=(10, 6))

# Iterate through each convolutional layer
for i, conv_layer in enumerate(conv_layers):
    # Extract weights of the current convolutional layer
    weights = conv_layer.weight.data.cpu().numpy()
    # Plot the first 5 filters in the convolutional layer
    for j in range(5):
        axs[i, j].imshow(weights[j, 0])
        axs[i, j].set_title(f'Filter {j+1}')
        axs[i, j].axis('off')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()



example_image, _ = train_set[0]
example_image = example_image.squeeze().numpy()

# Plot the original image
plt.figure(figsize=(10, 6))
plt.subplot(len(conv_layers)+1, 6, 1)
plt.imshow(example_image)
plt.title('Original Image')
plt.axis('off')

# Iterate through each convolutional layer
for i, conv_layer in enumerate(conv_layers):
    # Extract weights of the current convolutional layer
    weights = conv_layer.weight.data.cpu().numpy()
    # Iterate through each filter in the convolutional layer
    for j in range(min(weights.shape[0], 5)):  # Limit to 5 filters per layer
        # Get the filter weights
        filter_weights = weights[j, 0]
        # Apply the filter to the example image
        filtered_image = cv2.filter2D(example_image, -1, filter_weights)
        # Plot the filtered image
        plt.subplot(len(conv_layers)+1, 6, i*6 + j + 2)
        plt.imshow(filtered_image)
        plt.title(f'Layer {i+1},Filter {j+1}')
        plt.axis('off')

plt.tight_layout()
plt.show()

