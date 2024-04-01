"""
@author: Harshit Kumar, Khushi Neema
@brief: Utility functions for training and testing the model
"""

import torch

def train_test_network(model, train_loader, test_loader, optimizer, criterion, device, epochs=1):
    """
    Train and test the model for a number of epochs.

    Parameters:
    model (nn.Module): PyTorch model to train
    train_loader (DataLoader): DataLoader for training data
    test_loader (DataLoader): DataLoader for testing data
    optimizer (torch.optim.Optimizer): Optimizer for the model
    criterion (torch.nn.Module): Loss function
    epochs (int): Number of epochs to train the model

    Returns:
    list: Training losses for each epoch
    list: Testing losses for each epoch
    """
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_train += (predicted == target).sum().item()
            total_train += target.size(0)
        train_losses.append(running_loss / len(train_loader))
        train_acc.append(correct_train / total_train)
        
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output, 1)
                correct_test += (predicted == target).sum().item()
                total_test += target.size(0)
        test_losses.append(test_loss / len(test_loader))
        test_acc.append(correct_test / total_test)
        
        print(f"Epoch {epoch}/{epochs}, "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, "
              f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}")
    
    return train_losses, test_losses, train_acc, test_acc

def train_network(model, loader, optimizer, criterion, device, epochs=10):
    """
    Train the model for a number of epochs.

    Parameters:
    model (nn.Module): PyTorch model to train
    loader (DataLoader): DataLoader for training data
    optimizer (torch.optim.Optimizer): Optimizer for the model
    criterion (torch.nn.Module): Loss function
    epochs (int): Number of epochs to train the model

    Returns:
    list: Training losses for each epoch
    list: Training accuracies for each epoch
    """
    train_losses = []
    train_acc = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        train_losses.append(running_loss / len(loader))
        train_acc.append(correct_train / total_train)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}")

    return train_losses, train_acc