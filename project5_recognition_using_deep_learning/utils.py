import torch

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
