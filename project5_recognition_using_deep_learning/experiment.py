"""
@author: Khushi Neema, Harshit Kumar
@brief: Run an experiment to compare different configurations of a CNN on the Fashion MNIST dataset.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import pickle

from models.FashionMnistCNN import FashionMnistCNN
from utils import train_test_network

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Experiment configurations
batch_sizes = [32, 64, 128, 256, 512]
activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
conv_layers = [1, 2, 4, 6]
num_filters = [16, 32, 64, 128, 256]
dropout_rates = [0.0, 0.2, 0.5, 0.8]

# check for gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# dictionary to store results
results = {}

def evaluate_batch_sizes(train_set, test_set, batch_sizes, transform):
    """
    Evaluate different batch sizes on the given dataset.

    Args:
    - train_set: Dataset to train the model on
    - test_set: Dataset to test the model on
    - batch_sizes: List of batch sizes to evaluate
    - transform: Transformation to apply to the images

    Returns:
    - best_batch_size: The batch size that resulted in the highest test accuracy
    """
    global results  # To update the global results dictionary
    best_acc = 0
    best_batch_size = None
    
    for batch_size in batch_sizes:
        print(f"Evaluating batch size: {batch_size}")
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        model = FashionMnistCNN(conv_layers=2, num_filters=32, dropout_rate=0.5, activation='relu')
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        train_losses, test_losses, train_acc, test_acc = train_test_network(model, train_loader, test_loader, optimizer, criterion, device)
        
        results[(batch_size, 'relu')] = {'test_acc': test_acc[-1], 'train_acc': train_acc[-1], 'train_loss': train_losses[-1], 'test_loss': test_losses[-1]}
        
        last_test_acc = test_acc[-1]
        if last_test_acc > best_acc:
            best_acc = last_test_acc
            best_batch_size = batch_size
    
    print(f"Best batch size: {best_batch_size} with accuracy: {best_acc}")
    return best_batch_size

def evaluate_activation_functions(train_set, test_set, activations, best_batch_size, transform):
    """
    Evaluate different activation functions on the given dataset.

    Args:
    - train_set: Dataset to train the model on
    - test_set: Dataset to test the model on
    - activations: List of activation functions to evaluate
    - best_batch_size: The batch size that resulted in the highest test accuracy
    - transform: Transformation to apply to the images

    Returns:
    - best_activation: The activation function that resulted in the highest test accuracy
    """
    best_acc = 0
    best_activation = None
    
    for activation in activations:
        print(f"Evaluating activation function: {activation}")
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=best_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=best_batch_size, shuffle=False)
        
        model = FashionMnistCNN(conv_layers=2, num_filters=32, dropout_rate=0.5, activation=activation)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        train_losses, test_losses, train_acc, test_acc = train_test_network(model, train_loader, test_loader, optimizer, criterion, device)
        
        results[(best_batch_size, activation)] = {'test_acc': test_acc[-1], 'train_acc': train_acc[-1], 'train_loss': train_losses[-1], 'test_loss': test_losses[-1]}
        
        last_test_acc = test_acc[-1]
        if last_test_acc > best_acc:
            best_acc = last_test_acc
            best_activation = activation
    
    print(f"Best activation function: {best_activation} with accuracy: {best_acc}")
    return best_activation

def grid_search(train_set, test_set, conv_layers, num_filters, dropout_rates, best_batch_size, best_activation, transform):
    """
    Apply grid search to find the best combination of hyperparameters.

    Args:
    - train_set: Dataset to train the model on
    - test_set: Dataset to test the model on
    - conv_layers: List of convolutional layers to evaluate
    - num_filters: List of number of filters to evaluate
    - dropout_rates: List of dropout rates to evaluate
    - best_batch_size: The batch size that resulted in the highest test accuracy
    - best_activation: The activation function that resulted in the highest test accuracy
    - transform: Transformation to apply to the images

    Returns:
    - None
    """
    count_current_exp = 0
    count_total_exp = len(conv_layers) * len(num_filters) * len(dropout_rates)
    for conv_layer in conv_layers:
        for num_filter in num_filters:
            for dropout_rate in dropout_rates:
                count_current_exp += 1

                config = (conv_layer, num_filter, dropout_rate, best_batch_size, best_activation)
                print(f"Experiment {count_current_exp}/{count_total_exp} with config: {config}") 

                train_loader = torch.utils.data.DataLoader(train_set, batch_size=best_batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=best_batch_size, shuffle=False)
                
                model = FashionMnistCNN(conv_layers=conv_layer, num_filters=num_filter, dropout_rate=dropout_rate, activation=best_activation)
                model.to(device)
                optimizer = optim.AdamW(model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                
                train_losses, test_losses, train_acc, test_acc = train_test_network(model, train_loader, test_loader, optimizer, criterion, device)
                
                results[config] = {'test_acc': test_acc[-1], 'train_acc': train_acc[-1], 'train_loss': train_losses[-1], 'test_loss': test_losses[-1]}


# Step 1: Evaluate batch sizes
print("Evaluating batch sizes...")
best_batch_size = evaluate_batch_sizes(train_set, test_set, batch_sizes, transform)

# Step 2: Evaluate activation functions with the best batch size
print(f"Evaluating activation functions with best batch size: {best_batch_size}...")
best_activation = evaluate_activation_functions(train_set, test_set, activations, best_batch_size, transform)

# Step 3: Grid search on remaining parameters with the best batch size and activation function
print(f"Performing grid search with best batch size: {best_batch_size} and best activation function: {best_activation}...")
grid_search(train_set, test_set, conv_layers, num_filters, dropout_rates, best_batch_size, best_activation, transform)

# Functions used in the steps above would be defined to conduct the respective evaluations
# These should initialize the model with the given parameters, train it, and evaluate its performance
# based on the dataset splits. You'd then select the best performing parameters based on your criteria
# (e.g., highest accuracy, shortest training time, etc.).

# get best model from results
best_model = max(results, key=lambda x: results[x]['test_acc'])

# Save the comprehensive results to a file for later analysis
print('Saving results to model_config_exp.pkl')
with open('model_config_exp.pkl', 'wb') as f:
    pickle.dump(results, f)
