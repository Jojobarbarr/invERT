import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# MLP model to generate kernels for 3 convolutional layers
class KernelGeneratorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_sizes: int, num_kernels: int):
        super(KernelGeneratorMLP, self).__init__()
        self.input_channels: list[int] = [1] + num_kernels[:-1]  # Number of input channels for each layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, sum(k * k * c * n for k, c, n in zip(kernel_sizes, self.input_channels, num_kernels)))  # Combined output for all conv layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Fully convolutional network with 3 layers
class DynamicConvNet(nn.Module):
    def __init__(self, in_channels: int, num_kernels: list[int], kernel_sizes: list[int]):
        super(DynamicConvNet, self).__init__()
        self.kernel_sizes: list[int] = kernel_sizes
        self.in_channels: int = in_channels
        self.input_channels: list[int] = [1] + num_kernels[:-1]
        self.num_kernels: list[int] = num_kernels

        # Define convolutional layers without weights (they will be set dynamically)
        self.conv1 = nn.Conv2d(in_channels, num_kernels[0], kernel_size=kernel_sizes[0], bias=False, padding=1)
        self.conv2 = nn.Conv2d(num_kernels[0], num_kernels[1], kernel_size=kernel_sizes[1], bias=False, padding=1)
        self.conv3 = nn.Conv2d(num_kernels[1], 1, kernel_size=kernel_sizes[2], bias=False, padding=1)  # Output 1 channel


    def forward(self, x, kernels):
        # Split kernels for the three layers
        idx1 = self.num_kernels[0] * self.input_channels[0] * self.kernel_sizes[0] * self.kernel_sizes[0]
        idx2 = idx1 + self.num_kernels[1] * self.input_channels[1] * self.kernel_sizes[1] * self.kernel_sizes[1]

        kernels1 = kernels[:, :idx1].view(self.num_kernels[0], self.input_channels[0], self.kernel_sizes[0], self.kernel_sizes[0])
        kernels2 = kernels[:, idx1:idx2].view(self.num_kernels[1], self.input_channels[1], self.kernel_sizes[1], self.kernel_sizes[1])
        kernels3 = kernels[:, idx2:].view(self.num_kernels[2], self.input_channels[2], self.kernel_sizes[2], self.kernel_sizes[2])

        x = F.conv2d(x, kernels1, padding=1)
        x = F.tanh(x)
        x = F.conv2d(x, kernels2, padding=1)
        x = F.tanh(x)
        x = F.conv2d(x, kernels3, padding=1)

        return x

if __name__ == "__main__":
    # Example usage
    input_dim = 2  # MLP input size
    hidden_dim = 64
    kernel_sizes = [3, 3, 3]  # Kernel sizes for the three conv layers
    num_kernels = [16, 32, 1]  # Number of kernels (channels) for each layer; last layer outputs 1 channel

    # Initialize the models
    mlp = KernelGeneratorMLP(input_dim, hidden_dim, kernel_sizes, num_kernels)
    conv_net = DynamicConvNet(in_channels=1, num_kernels=num_kernels, kernel_sizes=kernel_sizes)
    print(f"conv_net: {conv_net}")

    # Input data
    mlp_input = torch.randn(1, input_dim)  # Input for MLP
    conv_input = torch.randn(1, 1, 28, 28)  # Example 2D input for the convolutional network

    # Forward pass through MLP to get the kernels
    generated_kernels = mlp(mlp_input)

    # Set the generated kernels as weights for the convolutional network

    # Forward pass through the convolutional network
    output = conv_net(conv_input, generated_kernels)

    print(f"conv_input.shape: {conv_input.shape}")
    print(f"output.shape: {output.shape}")  # The output will have 1 channel

    plt.imshow(conv_input[0, 0].detach().numpy(), cmap='gray')
    plt.show()
    plt.imshow(output[0, 0].detach().numpy(), cmap='gray')
    plt.show()