"""
@author: Khushi Neema, Harshit Kumar
@brief: Train and test a simple neural network on the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from models.MyNetwork import MyNetwork
from utils import train_test_network

def main():
    # Set up MNIST dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)
    

    '''
    A: Getting MNIST dataset
    '''
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    # Extract the first six examples from the test set
    first_six_examples = [test_dataset[i][0] for i in range(6)]

    # Plot the first six examples in a grid
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))

    for i, example in enumerate(first_six_examples):
        axes[i].imshow(example, cmap='gray')
        axes[i].set_title(f'Label: {test_dataset[i][1]}')
        axes[i].axis('off')

    plt.show()
    # Initialize network, optimizer, and criterion
    model = MyNetwork()

    if(0):
        #To see the diagram of the model architecture
        model = MyNetwork()
        dummy_input = torch.randn(1, 1, 28, 28)
        writer = SummaryWriter()
        writer.add_graph(model, dummy_input)
        writer.close()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train the network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_losses, test_losses, train_acc, test_acc = train_test_network(model, train_loader, test_loader, optimizer, criterion, device)
    
    # Plot training and testing errors
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()
    
    # Save the trained model
    torch.save(model.state_dict(), 'mnist_model.pth')

if __name__ == "__main__":
    main()