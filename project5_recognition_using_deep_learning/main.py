import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.log_softmax(self.fc2(x), dim=1)
        return x

# Useful functions
def train_network(train_loader, test_loader, model, optimizer, criterion, epochs=5):
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
    
    # Initialize network, optimizer, and criterion
    model = MyNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train the network
    train_losses, test_losses, train_acc, test_acc = train_network(train_loader, test_loader, model, optimizer, criterion)
    
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