import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# Fully connected network to generate the kernels
class KernelGeneratorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list[int], num_kernels: list[int], kernel_sizes: list[int]):
        super(KernelGeneratorMLP, self).__init__()

        # Define the fully connected layers
        self.fc_list = nn.ModuleList()
        self.fc_first = nn.Linear(input_dim, hidden_dim[0])
        for i in range(len(hidden_dim) - 1):
            self.fc_list.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
        self.fc_last = nn.Linear(hidden_dim[-1], sum(k * k * c * n for k, c, n in zip(kernel_sizes, [1] + num_kernels[:-1], num_kernels)))  # Combined output for all conv layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc_first(x))
        for fc in self.fc_list:
            x = F.relu(fc(x))
        x = self.fc_last(x)
        return x

# Fully convolutional network
class DynamicConvNet(nn.Module):
    def __init__(self, in_channels: int, num_kernels: list[int], kernel_sizes: list[int]):
        super(DynamicConvNet, self).__init__()
        self.in_channels: int = in_channels
        self.num_kernels: list[int] = num_kernels
        self.kernel_sizes: list[int] = kernel_sizes

    def forward(self, x, kernels):
        idx_list = []
        for layer_index in range(len(self.num_kernels)):
            if layer_index == 0:
                idx_list.append(self.num_kernels[layer_index] * self.kernel_sizes[layer_index] * self.kernel_sizes[layer_index])
            else:
                idx_list.append(idx_list[layer_index - 1] + self.num_kernels[layer_index] * self.num_kernels[layer_index-1] * self.kernel_sizes[layer_index] * self.kernel_sizes[layer_index])

        kernel_init = kernels[:, :idx_list[0]].view(self.num_kernels[0], 1, self.kernel_sizes[0], self.kernel_sizes[0])
        kernels_list = [kernel_init] + [kernels[:, idx_list[i-1]:idx_list[i]].view(self.num_kernels[i], self.num_kernels[i-1], self.kernel_sizes[i], self.kernel_sizes[i]) for i in range(1, len(self.num_kernels))]

        # Forward pass through the convolutional layers
        x = F.conv2d(x, kernels_list[0], padding=1)
        x = F.relu(x)
        for i in range(1, len(kernels_list) - 1):
            x = F.conv2d(x, kernels_list[i], padding=1)
            x = F.relu(x)
        x = F.conv2d(x, kernels_list[-1], padding=1)
        
        return x

class DynamicModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list[int], num_kernels: list[int], kernel_sizes: list[int], in_channels: int):
        super(DynamicModel, self).__init__()
        self.kernel_generator = KernelGeneratorMLP(input_dim, hidden_dim, num_kernels, kernel_sizes)
        self.conv_net = DynamicConvNet(in_channels, num_kernels, kernel_sizes)

    def forward(self, x_meatadata: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        kernels = self.kernel_generator(x_meatadata)
        return self.conv_net(x, kernels)

if __name__ == "__main__":
    # Example usage
    input_dim: int = 2  # MLP input size
    hidden_dim: list[int] = [64, 128]  # MLP hidden layers size
    kernel_sizes: list[int] = [3, 3, 3]  # CNN kernel sizes
    num_kernels: list[int] = [16, 32, 1]  # CNN number of kernels
    assert len(kernel_sizes) == len(num_kernels)
    assert num_kernels[-1] == 1

    # Initialize the models
    model = DynamicModel(input_dim, hidden_dim, num_kernels, kernel_sizes, 1)
    print(f"Model: {model}")

    # Input data
    input_data = torch.randn(1, 2)  # Metadata for the MLP
    input_image = torch.randn(1, 1, 28, 28)  # Example 2D input for the convolutional network)

    # Forward pass through MLP to get the kernels
    output = model(input_data, input_image)

    print(f"output.shape: {output.shape}")  # The output will have 1 channel

    plt.imshow(input_image[0, 0].detach().numpy(), cmap='gray')
    plt.show()
    plt.imshow(output[0, 0].detach().numpy(), cmap='gray')
    plt.show()