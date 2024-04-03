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


def predict_single_image(image_path, transform):
    """
    Predict the class of a single image.

    Args:
    image_path (str): Path to the image file.
    transform (torchvision.transforms.Compose): Image transformation to apply.
    """
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(output, dim=1)
    print(f"Network output probabilities: {probabilities.squeeze().detach().numpy().round(2)}")
    print(f"Predicted class: {predicted_class.item()}, Actual class: {int(image_path.split('/')[-1][0])}")  # Assuming the label is part of the filename
    return int(image_path.split('/')[-1][0])

def display_predictions(image_paths, transform):
    """
    Display predictions on new handwritten digit images.

    Args:
    image_paths (list): List of paths to the image files.
    transform (torchvision.transforms.Compose): Image transformation to apply.
    """
    plt.figure(figsize=(10, 6))
    for i, image_path in enumerate(image_paths, 1):
        if i>10:
            break
        image = Image.open(image_path)
        transformed_image = transform(image)
        prediction = predict_single_image(image_path, transform)
        print(prediction)
        if i<10:
            plt.subplot(3, 3, i)  # Ensure that the third argument is within the range [1, 9]
            plt.imshow(transformed_image.squeeze().numpy(), cmap='gray')
            plt.title(f'Prediction: {prediction}')
            plt.axis('off')
    plt.tight_layout()
    plt.show()


# Test the model on the MNIST test set
def test_on_mnist_test_set():
    plt.figure(figsize=(10,6))
    mnist_test_set = MNIST(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(mnist_test_set, batch_size=1, shuffle=False)
    
    # Iterate over the first 10 examples in the test set
    for i, (image, label) in enumerate(test_loader,1):
        if i > 10:
            break
        output = model(image)
        #_, predicted = torch.max(output, 1)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1)
        print(f"Network output probabilities: {probabilities.squeeze().detach().numpy().round(2)}")
        print(f"Predicted class: {predicted_class.item()}, Actual class: {label.item()}")
        if i<10:
            plt.subplot(3,3,i)
            plt.imshow(image.squeeze().numpy(), cmap='grey')
            plt.title(f'Prediction: {label.item()}')
            plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define transformations for test set images
    transform_test = transforms.Compose([
        #transforms.Grayscale(),
        # transforms.Resize((28, 28)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: 1 - x),  # Invert pixel intensities
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Define transformations for new handwritten digit images
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        # transforms.Lambda(lambda img: img.rotate(-90)),
        #transforms.RandomHorizontalFlip(),  # Horizontally flip with 50% probability
        transforms.ToTensor(),
        # transforms.RandomRotation(35),
        # transforms.functional.rotate(angle=90),
        # transforms.Lambda(lambda x: 1 - x),  # Invert pixel intensities
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()

    # Paths to new handwritten digit images
    new_images_paths = glob.glob('./data/images/*.jpeg')

    # Display predictions on new handwritten digit images
    display_predictions(new_images_paths[:-1], transform)

    # Test the model on the MNIST test set
    test_on_mnist_test_set()
