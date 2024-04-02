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
import numpy as np
from models.MyNetwork import MyNetwork

# Load the trained model
model = MyNetwork()
model.load_state_dict(torch.load('greek_model.pth'))
model.eval()

class GreekTransform:
    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale(x)
        x = transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = transforms.functional.center_crop(x, (28, 28))
        return transforms.functional.invert(x)

def predict_single_image(image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()


def display_predictions(image_paths, transform):
    plt.figure(figsize=(10, 6))
    for i, image_path in enumerate(image_paths, 1):
        if i>=10:
            break
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

# Paths to new handwritten greek images
new_images_paths = glob.glob('./data/greek_images/*.png')

# Display predictions on new handwritten digit images
display_predictions(new_images_paths, GreekTransform())

# Test the model on the MNIST test set
test_on_mnist_test_set()