import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# Fully connected network to generate the kernels


class KernelGeneratorMLP(nn.Module):
    def __init__(self,
                 input_metadata_dim: int,
                 hidden_dims: list[int],
                 in_channels: list[int],
                 out_channels: list[int],
                 kernel_shapes: list[int]
                 ) -> None:
        super(KernelGeneratorMLP, self).__init__()

        self.input_metadata_dim: int = input_metadata_dim
        self.hidden_dims: list[int] = hidden_dims
        self.in_channels: list[int] = in_channels
        self.out_channels: list[int] = out_channels
        self.kernel_shapes: list[int] = kernel_shapes

        self.nbr_weights_layers: list[int] = [
            in_channels[i]
            * out_channels[i]
            * kernel_shapes[i]
            * kernel_shapes[i]
            for i in range(len(in_channels))
        ]

        self.total_nbr_weights: int = sum(self.nbr_weights_layers)

        self.filter_generator = nn.Sequential(
            nn.ModuleList(
                [nn.Linear(self.input_metadata_dim, self.hidden_dims[0])]
                + [nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
                   for i in range(len(self.hidden_dims) - 1)]
            ),
            nn.Linear(self.hidden_dims[-1], self.total_nbr_weights)
        )

    def forward(self,
                x: torch.Tensor
                ) -> list[torch.Tensor]:
        """
        Forward pass through the network.

        @param x: The input tensor of shape (batch_size, input_dim)
        @return: A list of tensors (one for each convolutionnal layer) of
        shape (batch_size, nbr_in_channels, nbr_out_channels,
        kernel_size, kernel_size)
        """
        batch_size: int = x.shape[0]
        x = self.filter_generator(x)
        output_list: list[torch.Tensor] = []
        start_idx: int = 0
        for layer in len(self.in_channels):
            end_idx: int = start_idx + self.nbr_weights_layers[layer]
            output_list.append(
                x[:, start_idx:end_idx].view(
                    batch_size,
                    self.in_channels[layer],
                    self.out_channels[layer],
                    self.kernel_shapes[layer],
                    self.kernel_shapes[layer]
                )
            )
            output_list[-1] = output_list[-1].view(
                1,
                batch_size * self.in_channels[layer],
                self.kernel_shapes[layer],
                self.kernel_shapes[layer]
            )
            start_idx = end_idx
        print(f"output_list[0].shape: {output_list[0].shape}")
        return output_list

# Fully convolutional network


class DynamicConvNet(nn.Module):
    def __init__(
            self,
            input_nbr_channels: int,
            nbr_kernels: list[int],
            kernel_shapes: list[int]):
        super(DynamicConvNet, self).__init__()

        self.input_nbr_channels: int = input_nbr_channels  # Useless?
        self.nbr_kernels: list[int] = nbr_kernels
        self.kernel_shapes: list[int] = kernel_shapes
        self.bn_list = nn.ModuleList(
            [nn.BatchNorm2d(nbr_kernel) for nbr_kernel in self.nbr_kernels]
        )

    def forward(self,
                x: torch.Tensor,
                kernels: list[torch.Tensor]
                ) -> torch.Tensor:
        """
        """
        for kernel in kernels:
            x = F.conv2d(x, kernel, padding="same")



        idx_list = []
        for layer_index in range(len(self.nbr_kernels)):
            if layer_index == 0:
                idx_list.append(
                    self.nbr_kernels[layer_index]
                    * self.kernel_shapes[layer_index]
                    * self.kernel_shapes[layer_index])
            else:
                idx_list.append(idx_list[layer_index - 1]
                                + self.nbr_kernels[layer_index]
                                * self.nbr_kernels[layer_index - 1]
                                * self.kernel_shapes[layer_index]
                                * self.kernel_shapes[layer_index])
        print(f"x.shape: {x.shape}")
        print(f"kernels.shape: {kernels.shape}")
        kernel_init = kernels[:, :idx_list[0]].view(
            kernels.shape[0],
            self.nbr_kernels[0],
            1,
            self.kernel_shapes[0],
            self.kernel_shapes[0])
        print(f"kernel_init.shape: {kernel_init.shape}")
        kernels_list = [kernels[:, idx_list[i - 1]:idx_list[i]].view(
            kernels.shape[0],
            self.nbr_kernels[i],
            self.nbr_kernels[i - 1],
            self.kernel_shapes[i],
            self.kernel_shapes[i]) for i in range(1, len(self.nbr_kernels))]
        print(f"kernels_list[0].shape: {kernels_list[0].shape}")
        kernels_list = [kernel_init] + kernels_list

        # Forward pass through the convolutional layers
        print(f"x.shape: {x.shape}")
        print(f"kernels_list[0].shape: {kernels_list[0].shape}")
        x = F.conv2d(x, kernels_list[0], padding="same")
        # x = F.relu(x)
        x = F.relu(self.bn_list[0](x))
        for i in range(1, len(kernels_list) - 1):
            x = F.conv2d(x, kernels_list[i], padding="same")
            # x = F.relu(x)
            x = F.relu(self.bn_list[i](x))
        x = F.conv2d(x, kernels_list[-1], padding="same")
        # x = F.relu(x)
        # x = F.relu(self.bn_list[-1](x))
        x = F.tanh(x)
        # x = F.tanh(self.bn_list[-1](x))
        return x


class DynamicModel(nn.Module):
    def __init__(
            self,
            input_metadata_dim: int,
            hidden_dims: list[int],
            in_channels: list[int],
            out_channels: list[int],
            kernel_shapes: list[int]):
        """
        :param input_dim: The input dimension of the metadata
        :param hidden_dim: The hidden layer sizes for the MLP
        :param nbr_kernels: The number of kernels for each convolutional layer
        :param kernel_shapes: The kernel sizes for each convolutional layer
        :param in_channels: The number of input channels for the
                             convolutional network

        """
        super(DynamicModel, self).__init__()
        self.kernel_generator = KernelGeneratorMLP(
            input_metadata_dim,
            hidden_dims,
            in_channels,
            out_channels,
            kernel_shapes
        )
        self.conv_net = DynamicConvNet(in_channels, nbr_kernels, kernel_shapes)

    def forward(self, x_meatadata: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        kernels = self.kernel_generator(x_meatadata)
        x = self.conv_net(x, kernels)
        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


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
    # Example 2D input for the convolutional network)
    input_image = torch.randn(1, 1, 28, 28)

    # Forward pass through MLP to get the kernels
    output = model(input_data, input_image)

    print(f"output.shape: {output.shape}")  # The output will have 1 channel

    plt.imshow(input_image[0, 0].detach().numpy(), cmap='gray')
    plt.show()
    plt.imshow(output[0, 0].detach().numpy(), cmap='gray')
    plt.show()
