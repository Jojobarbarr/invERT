import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelGeneratorMLP(nn.Module):
    def __init__(self,
                 input_metadata_dim: int,
                 hidden_dims: list[int],
                 layer_types: list[str],
                 num_in_channels: list[int],
                 kernel_shapes: list[tuple[int]],
                 num_out_channels: list[int],
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
        self.layer_type: list[str] = layer_types
        self.num_in_channels: list[int] = num_in_channels
        self.kernel_shapes: list[tuple[int]] = kernel_shapes
        self.num_out_channels: list[int] = num_out_channels
        self.num_conv_layers: int = len(num_in_channels)

        # Compute the number of weights needed for each conv layer
        self.nbr_weights_conv_layers = [
            num_in_channels[i] * num_out_channels[i] * (kernel_shapes[i][0] * kernel_shapes[i][1])
            for i in range(self.num_conv_layers)
        ]

        self.total_nbr_weights: int = sum(self.nbr_weights_conv_layers)

        print(f"Total number of weights for CNN: {self.total_nbr_weights}")

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
        shape (num_out_channels, num_in_channels, kernel_size, kernel_size)
        """
        # Pass through the MLP to get a flat vector of kernel weights
        x = self.mlp(x)

        output_list: list[torch.Tensor] = []
        start_idx: int = 0
        for layer, layer_type in zip(range(self.num_conv_layers), self.layer_type):
            end_idx = start_idx + self.nbr_weights_conv_layers[layer]
            if layer_type == "conv":
                kernel = x[:, start_idx:end_idx].reshape(
                    self.num_out_channels[layer],
                    self.num_in_channels[layer],
                    self.kernel_shapes[layer][0],
                    self.kernel_shapes[layer][1]
                )
            elif layer_type == "transpose_conv":
                kernel = x[:, start_idx:end_idx].reshape(
                    self.num_in_channels[layer],
                    self.num_out_channels[layer],
                    self.kernel_shapes[layer][0],
                    self.kernel_shapes[layer][1]
                )
            output_list.append(kernel)
            start_idx = end_idx

        return output_list


class DynamicConv2D(nn.Module):
    def __init__(self, stride: tuple[int], padding: tuple[int]) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(
            x,
            kernels,
            stride=self.stride,
            padding=self.padding
        )
        return x

class DynamicConvTranspose2D(nn.Module):
    def __init__(self, stride: tuple[int], padding: tuple[int]) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        x = F.conv_transpose2d(
            x,
            kernels,
            stride=self.stride,
            padding=self.padding,
        )
        return x


class DynamicCNN(nn.Module):
    def __init__(self,
                 layer_types: list[str],
                 kernel_shapes: list[tuple[int]],
                 strides: list[tuple[int]],
                 paddings: list[tuple[int]],
                 num_out_channel: list[int]
                 ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "conv":
                self.layers.append(DynamicConv2D(strides[layer_idx], paddings[layer_idx]))
            elif layer_type == "transpose_conv":
                self.layers.append(DynamicConvTranspose2D(strides[layer_idx], paddings[layer_idx]))
            elif layer_type == "maxpool":
                self.layers.append(nn.MaxPool2d(kernel_size=kernel_shapes[layer_idx], stride=strides[layer_idx]))
            elif layer_type == "batchnorm":
                self.layers.append(nn.BatchNorm2d(num_features=num_out_channel[layer_idx]))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

    def forward(self,
                x: torch.Tensor,
                kernels: list[torch.Tensor],
                target: torch.Tensor
                ) -> torch.Tensor:

        for layer_idx, (layer, layer_type) in enumerate(zip(self.dynamic_layers[:-1], self.layer_types)):
            if layer_type == "conv" or layer_type == "transpose_conv":
                x = layer(x, kernels[layer_idx])
                x = F.relu(x)
            elif layer_type == "maxpool":
                x = layer(x)
            elif layer_type == "batchnorm":
                x = layer(x)
        x = F.interpolate(x, size=(target.shape[2], target.shape[3]),
                            mode="bilinear", align_corners=False)
        x = F.sigmoid(self.dynamic_layers[-1](x, kernels[-1]))
        return x


class DynamicModel(nn.Module):
    def __init__(self,
                 input_metadata_dim: int,
                 hidden_dims: list[int],
                 layer_types: list[str],
                 num_in_channels: list[int],
                 kernel_shapes: list[tuple[int]],
                 strides: list[tuple[int]],
                 paddings: list[tuple[int]],
                 num_out_channels: int,
    ) -> None:
        """
        The complete dynamic model consists of:
          - A kernel generator network (an MLP) that produces convolutional
          kernels from metadata.
          - A fully convolutional network that uses these kernels.

        num_in_channels: List of input channels for each convolutional layer.
        num_out_channels: The number of output channels of the final layer.
        The effective conv channels (per layer) are set to:
           [num_in_channels[1], num_in_channels[2], ..., num_out_channels]
        """
        super().__init__()

        self.num_in_channels: list[int] = num_in_channels
        # Construct a list of output channels for each conv layer.
        # For a single-layer conv net, the only layer maps
        # num_in_channels[0] -> num_out_channel.
        if len(num_in_channels) > 1:
            self.conv_num_out_channels: list[int] = \
                num_in_channels[1:] + [num_out_channels]
        else:
            self.conv_num_out_channels: list[int] = [num_out_channels]

        self.kernel_generator = KernelGeneratorMLP(
            input_metadata_dim,
            hidden_dims,
            layer_types,
            self.num_in_channels,
            kernel_shapes,
            self.conv_num_out_channels,
        )

        self.conv_net = DynamicCNN(
            layer_types,
            kernel_shapes,
            strides,
            paddings,
            self.conv_num_out_channels
        )

    def forward(self,
                x_metadata: torch.Tensor,
                x: torch.Tensor,
                target: torch.Tensor
                ) -> torch.Tensor:
        """
        x_metadata: Tensor of shape (batch_size, input_metadata_dim)
        x: Tensor of shape (batch_size, num_channels, height, width)
        """
        kernels = self.kernel_generator(x_metadata)
        num_channels: int = x.shape[1]
        # Reshape x so that group convolution can assign each batch element
        # its own kernel.
        x = x.view(1, num_channels, x.shape[2], x.shape[3])
        x = self.conv_net(x, kernels, target)
        return x.view(
            1,
            self.conv_num_out_channels[-1],
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
        input_metadata_dim=3,
        hidden_dims=[64, 128],
        layer_types=["conv", "conv", "transpose_conv", "transpose_conv", "conv"],
        num_in_channels=[1, 16, 16, 16, 16],
        kernel_shapes=[(3, 3), (3, 3), (3, 3), (5, 3), (3, 3)],
        strides=[(1, 1), (1, 1), (2, 2), (4, 2), (1, 1)],
        num_out_channel=1,
    )
    x_metadata = torch.rand((1, 3))

    x = torch.rand((1, 1, 28, 45))
    target = torch.rand((1, 1, 76, 153))
    y = model(x_metadata, x, target)
    print(y.shape)
