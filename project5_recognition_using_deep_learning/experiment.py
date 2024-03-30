import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import pickle

from FashionMnistCNN import FashionMnistCNN
from utils import train_network

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

# Experiment configurations
configs = [
    # (conv_layers, num_filters, dropout_rate)
    (1, 16, 0.2), 
    (2, 32, 0.4), 
    (2, 64, 0.5),
    (3, 64, 0.6), 
    (4, 128, 0.8),
]

# dictionary to store results
results = {} 

for i, config in enumerate(configs):
    print(f"Training model {i+1}/{len(configs)} with config: {config}")
    conv_layers, num_filters, dropout_rate = config
    # Define the model with current configuration
    model = FashionMnistCNN(conv_layers, num_filters, dropout_rate)
    # Train the model
    start_time = time.time()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_losses, test_losses, train_acc, test_acc = train_network(train_loader, test_loader, model, optimizer, criterion)
    training_time = time.time() - start_time
    # Record results
    results[config] = {'train_losses': train_losses, 'test_losses': test_losses, 'train_acc': train_acc, 'test_acc': test_acc, 'training_time': training_time}

# Export results dictionary to a file
print('Saving results to modl_config_exp.pkl')
with open('modl_config_exp.pkl', 'wb') as f:
    pickle.dump(results, f)
    
print(results)
