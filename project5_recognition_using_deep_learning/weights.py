"""
@author: Khushi Neema, Harshit Kumar
@brief: Visualize the filters of the first convolutional layer of a CNN trained on the MNIST dataset.
"""

import torch
from main import MyNetwork
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import cv2

model =MyNetwork()

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image or numpy.ndarray to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the image data
])

# Load the MNIST training dataset
train_set = MNIST(root='./data', train=True, download=True, transform=transform)


weight_firstlayer=model.conv1.weight

# print("Shape of conv1 weights:", weight_firstlayer.shape)
# for i in range(10):
#     print("Filter", i+1, "weights:", weight_firstlayer[i, 0])

# # Visualize the ten filters
# plt.figure(figsize=(10, 8))
# for i in range(10):
#     plt.subplot(3, 4, i+1)
#     plt.imshow(weight_firstlayer[i, 0].detach().numpy(), cmap='gray')
#     plt.title('Filter ' + str(i+1))
#     plt.xticks([])
#     plt.yticks([])
# plt.tight_layout()
# plt.show()



with torch.no_grad():
    # Get the first training example image
    example_image, _ = train_set[0]
    example_image = example_image.unsqueeze(0)  # Add batch dimension

    # Apply the ten filters to the image
    filtered_images = []
    for i in range(10):
        # Get the filter weights
        filter_weights = weight_firstlayer[i, 0].numpy()

        # Apply the filter using OpenCV's filter2D function
        filtered_image = cv2.filter2D(example_image.numpy()[0, 0], -1, filter_weights)

        # Normalize the filtered image to be in the range [0, 1]
        filtered_image = (filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min())

        # Append the filtered image to the list
        filtered_images.append(filtered_image)

# Generate a plot of the 10 filtered images
plt.figure(figsize=(10, 8))
for i in range(10):
    plt.subplot(3, 4, i+1)
    plt.imshow(filtered_images[i], cmap='gray')
    plt.title('Filtered Image ' + str(i+1))
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()