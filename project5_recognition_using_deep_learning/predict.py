"""
@author: Khushi Neema, Harshit Kumar
@brief: Predict on new handwritten digit images using a trained MNIST model.
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import glob

from models.MyNetwork import MyNetwork

# Load the trained model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Define transformations for new inputs
# transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: 1 - x),  # Invert pixel intensities
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    #transforms.RandomRotation(degrees=15),  # Rotate by +/- 15 degrees
    transforms.RandomHorizontalFlip(),  # Horizontally flip with 50% probability
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),  # Invert pixel intensities
    transforms.Normalize((0.1307,), (0.3081,))
])

# Function to predict on a single image
def predict_single_image(image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()


def display_predictions(image_paths, transform):
    plt.figure(figsize=(10, 6))
    for i, image_path in enumerate(image_paths, 1):
        image = Image.open(image_path)
        transformed_image = transform(image)
        prediction = predict_single_image(image_path, transform)
        print(prediction)
        plt.subplot(3, 3, i)  # Ensure that the third argument is within the range [1, 9]
        plt.imshow(transformed_image.squeeze().numpy(), cmap='gray')
        plt.title(f'Prediction: {prediction}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Test the model on the MNIST test set
def test_on_mnist_test_set():
    mnist_test_set = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_test_set, batch_size=1, shuffle=False)
    
    # Iterate over the first 10 examples in the test set
    for i, (image, label) in enumerate(test_loader):
        if i >= 10:
            break
        output = model(image)
        _, predicted = torch.max(output, 1)
        print(f"Example {i+1}: Predicted={predicted.item()}, Actual={label.item()}")

# Paths to new handwritten digit images
new_images_paths = glob.glob('./data/images/*.jpg')

# Display predictions on new handwritten digit images
display_predictions(new_images_paths[:-1], transform)

# Test the model on the MNIST test set
test_on_mnist_test_set()
