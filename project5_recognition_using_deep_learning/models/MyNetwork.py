import torch.nn as nn
import torch
# from torchviz import make_dot
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


architecture = [
    ("Conv2d(1, 10, kernel_size=5)", (28, 28)),
    ("MaxPool2d(kernel_size=2, stride=2)", (10, 12, 12)),
    ("Conv2d(10, 20, kernel_size=5)", (20, 8, 8)),
    ("Dropout(p=0.5)", (20, 8, 8)),
    ("MaxPool2d(kernel_size=2, stride=2)", (20, 4, 4)),
    ("Flatten()", (320,)),
    ("Linear(in_features=320, out_features=50)", (50,)),
    ("ReLU()", (50,)),
    ("Linear(in_features=50, out_features=10)", (10,)),
    ("LogSoftmax(dim=1)", (10,))
]

# Function to draw the architecture diagram
# def draw_architecture(architecture):
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # Draw rectangles for each layer
#     for i, (layer, shape) in enumerate(architecture):
#         rect = plt.Rectangle((i * 0.1, 0), 0.1, 1, color='skyblue', ec='black')
#         ax.add_patch(rect)
#         ax.text(i * 0.1 + 0.05, 0.5, layer, va='center', ha='center', rotation=90)

#         # Draw arrows between layers
#         if i < len(architecture) - 1:
#             plt.arrow(i * 0.1 + 0.1, 0.5, 0.1, 0, color='black', head_width=0.05)

#     ax.set_xlim(-0.05, len(architecture) * 0.1)
#     ax.set_ylim(0, 1)
#     ax.axis('off')

#     plt.title("Network Architecture")
#     plt.tight_layout()
#     plt.savefig("network_diagram.png")

# # Draw and save the architecture diagram
# draw_architecture(architecture)


