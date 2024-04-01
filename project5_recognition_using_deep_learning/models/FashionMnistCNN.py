import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionMnistCNN(nn.Module):
    def __init__(self, conv_layers=2, num_filters=32, dropout_rate=0.5, activation='relu'):
        super().__init__()
        self.conv_layers = conv_layers
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Dynamically create convolutional layers
        self.convs = nn.ModuleList()
        in_channels = 1
        for i in range(conv_layers):
            out_channels = num_filters * (2 ** i)
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        
        # Activation function mapping
        if activation == "relu":
            self.act_fn = nn.ReLU()
        elif activation == "tanh":
            self.act_fn = nn.Tanh()
        elif activation == "sigmoid":
            self.act_fn = nn.Sigmoid()
        elif activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Adaptive pooling and fully connected layers
        self.adap_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(out_channels * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        for conv in self.convs:
            x = self.act_fn(conv(x))
            x = F.max_pool2d(x, 2)
        x = self.adap_pool(x)
        x = torch.flatten(x, 1)
        x = self.act_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x