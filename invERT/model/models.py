import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelGeneratorMLP(nn.Module):
    def __init__(self,
                 input_metadata_dim: int,
                 hidden_dims: list[int],
                 in_channels: list[int],
                 out_channels: list[int],
                 kernel_shapes: list[int]
                 ) -> None:
        """
        Generates the kernels (flattened weights) for a sequence of
        convolutional layers.
        The final MLP outputs a vector that is sliced and reshaped into
        individual kernels.
        """
        super().__init__()

        self.input_metadata_dim: int = input_metadata_dim
        self.hidden_dims: list[int] = hidden_dims
        self.in_channels: list[int] = in_channels
        self.out_channels: list[int] = out_channels
        self.kernel_shapes: list[int] = kernel_shapes
        self.num_conv_layers: int = len(in_channels)

        # Compute the number of weights needed for each conv layer
        self.nbr_weights_conv_layers = [
            in_channels[i] * out_channels[i] * (kernel_shapes[i] ** 2)
            for i in range(self.num_conv_layers)
        ]

        self.total_nbr_weights: int = sum(self.nbr_weights_conv_layers)

        # Build the MLP as a sequential module.
        layers = []
        layers.append(nn.Linear(input_metadata_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], self.total_nbr_weights))
        self.mlp = nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor
                ) -> list[torch.Tensor]:
        """
        Forward pass through the network.

        @param x: The input tensor of shape (batch_size, input_dim)
        @return: A list of tensors (one for each convolutionnal layer) of
        shape (batch_size * out_channels, nbr_in_channels,
        kernel_size, kernel_size)
        """
        batch_size: int = x.shape[0]
        # Pass through the MLP to get a flat vector of kernel weights
        x = self.mlp(x)

        output_list: list[torch.Tensor] = []
        start_idx: int = 0
        for layer in range(self.num_conv_layers):
            end_idx = start_idx + self.nbr_weights_conv_layers[layer]
            kernel = x[:, start_idx:end_idx].reshape(
                batch_size * self.out_channels[layer],
                self.in_channels[layer],
                self.kernel_shapes[layer],
                self.kernel_shapes[layer]
            )
            output_list.append(kernel)
            start_idx = end_idx

        return output_list


class DynamicConv2D(nn.Module):
    def __init__(self,
                 stride: int = 1,
                 padding: str = "same"
                 ) -> None:
        """
        A convolutional layer that takes dynamically generated kernels.
        """
        super().__init__()
        self.stride: int = stride
        self.padding: str = padding

    def forward(self,
                x: torch.Tensor,
                kernels: torch.Tensor,
                batch_size: int,
                ) -> torch.Tensor:
        # Use group convolution so that each image in the batch uses its own
        # kernel.
        x = F.conv2d(x,
                     kernels,
                     stride=self.stride,
                     padding=self.padding,
                     groups=batch_size)

        return x


class DynamicConvNet(nn.Module):
    def __init__(self,
                 in_channels: list[int],
                 ) -> None:
        """
        A network composed of several dynamic convolution layers.
        """
        super().__init__()

        self.num_layers = len(in_channels)
        self.dynamic_conv_layers = nn.ModuleList(
            [DynamicConv2D() for _ in range(self.num_layers)]
        )

    def forward(self,
                x: torch.Tensor,
                kernels: list[torch.Tensor],
                batch_size: int,
                ) -> torch.Tensor:
        for conv_layer, kernel in zip(self.dynamic_conv_layers, kernels):
            x = conv_layer(x, kernel, batch_size)
        return x


class DynamicModel(nn.Module):
    def __init__(self,
                 input_metadata_dim: int,
                 hidden_dims: list[int],
                 in_channels: list[int],
                 out_channels: int,
                 kernel_shapes: list[int]
                 ) -> None:
        """
        The complete dynamic model consists of:
          - A kernel generator network (an MLP) that produces convolutional
          kernels from metadata.
          - A fully convolutional network that uses these kernels.

        in_channels: List of input channels for each convolutional layer.
        out_channels: The number of output channels of the final layer.
        The effective conv channels (per layer) are set to:
           [in_channels[1], in_channels[2], ..., out_channels]
        """
        super().__init__()

        self.in_channels: list[int] = in_channels
        # Construct a list of output channels for each conv layer.
        # For a single-layer conv net, the only layer maps
        # in_channels[0] -> out_channels.
        if len(in_channels) > 1:
            self.conv_out_channels: list[int] = \
                in_channels[1:] + [out_channels]
        else:
            self.conv_out_channels: list[int] = [out_channels]

        self.kernel_generator = KernelGeneratorMLP(
            input_metadata_dim,
            hidden_dims,
            self.in_channels,
            self.conv_out_channels,
            kernel_shapes
        )

        self.conv_net = DynamicConvNet(self.in_channels)

    def forward(self,
                x_metadata: torch.Tensor,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        x_metadata: Tensor of shape (batch_size, input_metadata_dim)
        x: Tensor of shape (batch_size, num_channels, height, width)
        """
        kernels = self.kernel_generator(x_metadata)
        batch_size: int = x.shape[0]
        num_channels: int = x.shape[1]
        # Reshape x so that group convolution can assign each batch element
        # its own kernel.
        x = x.view(1, batch_size * num_channels, x.shape[2], x.shape[3])
        x = self.conv_net(x, kernels, batch_size)
        return x.view(
            batch_size,
            self.conv_out_channels[-1],
            x.shape[2],
            x.shape[3]
        )


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # Example usage
    model = DynamicModel(
        input_metadata_dim=2,
        hidden_dims=[64, 128],
        in_channels=[2, 16, 32],
        out_channel=3,
        kernel_shapes=[3, 3, 3]
    )
    x_metadata = torch.rand((16, 2))

    x = torch.rand((16, 2, 28, 45))
    y = model(x_metadata, x)
    print(y.shape)
