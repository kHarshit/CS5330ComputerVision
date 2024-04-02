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
import argparse
import glob
from PIL import Image

from models.MyNetwork import MyNetwork
from utils import train_network


# read command line arguments
parser = argparse.ArgumentParser(description='Train a simple neural network on the Greek dataset.')
parser.add_argument('--mode', type=str, default='train', help='Mode to run the script in: train or predict')

args = parser.parse_args()

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained MNIST model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
# model.eval()

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

transform = transforms.Compose([
                                transforms.Grayscale(),
                                transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                GreekTransform(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

def plot_metrics():
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

class_names = {0: 'alpha', 1: 'beta', 2: 'gamma'}

def predict_single_image(image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]  # Return the Greek class name
    # return predicted.item()


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


if args.mode == 'train':
    training_set_path = './data/greek_train/'

    greek_dataset = ImageFolder(root=training_set_path,
                                transform=transform)

    greek_train_loader = DataLoader(greek_dataset, batch_size=5, shuffle=True)

    optimizer = optim.Adam(model.fc2.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the network
    train_losses, train_acc = train_network(model, greek_train_loader, optimizer, criterion, device, 15)

    # Save the trained model
    torch.save(model.state_dict(), 'greek_model.pth')

    # plot the training loss and accuracy
    plot_metrics()

elif args.mode == 'predict':
    model.load_state_dict(torch.load('greek_model.pth'))
    model.eval()
    # Paths to new handwritten greek images
    new_images_paths = glob.glob('./data/greek_handdrawn/*.png')

    # Display predictions on new handwritten digit images
    display_predictions(new_images_paths, transform)