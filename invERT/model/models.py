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
        super(KernelGeneratorMLP, self).__init__()

        self.input_metadata_dim: int = input_metadata_dim
        self.hidden_dims: list[int] = hidden_dims
        self.in_channels: list[int] = in_channels
        self.out_channels: list[int] = out_channels
        self.kernel_shapes: list[int] = kernel_shapes
        self.num_conv_layers: int = len(in_channels)

        self.nbr_weights_conv_layers: list[int] = [
            self.in_channels[i]
            * self.out_channels[i]
            * self.kernel_shapes[i]
            * self.kernel_shapes[i]
            for i in range(len(in_channels))
        ]

        self.total_nbr_weights: int = sum(self.nbr_weights_conv_layers)

        self.layers: nn.ModuleList[nn.Linear] = nn.ModuleList([
            nn.Linear(self.input_metadata_dim, self.hidden_dims[0])
        ] + [
            nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
            for i in range(len(self.hidden_dims) - 1)
        ] + [nn.Linear(self.hidden_dims[-1], self.total_nbr_weights)])

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
        for layer in self.layers:
            x = F.relu(layer(x))
        output_list: list[torch.Tensor] = []
        start_idx: int = 0
        for layer in range(self.num_conv_layers):
            end_idx: int = start_idx + self.nbr_weights_conv_layers[layer]
            output_list.append(
                x[:, start_idx:end_idx].reshape(
                    self.out_channels[layer] * batch_size,
                    self.in_channels[layer],
                    self.kernel_shapes[layer],
                    self.kernel_shapes[layer]
                )
            )
            start_idx = end_idx
        return output_list


class DynamicConv2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_shape: int,
                 stride: int = 1,
                 padding: str = "same"
                 ) -> None:
        super(DynamicConv2D, self).__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_shape: int = kernel_shape
        self.stride: int = stride
        self.padding: str = padding

    def forward(self,
                x: torch.Tensor,
                kernels: torch.Tensor,
                batch_size: int,
                ) -> torch.Tensor:
        x = F.conv2d(x,
                     kernels,
                     stride=self.stride,
                     padding=self.padding,
                     groups=batch_size)

        return x


# Fully convolutional network
class DynamicConvNet(nn.Module):
    def __init__(self,
                 in_channels: list[int],
                 out_channels: list[int],
                 kernel_shapes: list[int]
                 ) -> None:
        super(DynamicConvNet, self).__init__()

        self.num_layers = len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_shapes = kernel_shapes

        self.dynamic_conv_layers = nn.ModuleList([
            DynamicConv2D(
                self.in_channels[i],
                self.out_channels[i],
                kernel_shapes[i]
            ) for i in range(self.num_layers)
        ])

    def forward(self,
                x: torch.Tensor,
                kernels: list[torch.Tensor],
                batch_size: int,
                ) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.dynamic_conv_layers[i](x, kernels[i], batch_size)
        return x


class DynamicModel(nn.Module):
    def __init__(self,
                 input_metadata_dim: int,
                 hidden_dims: list[int],
                 in_channels: list[int],
                 out_channel: int,
                 kernel_shapes: list[int]
                 ) -> None:
        super(DynamicModel, self).__init__()

        self.in_channels: list[int] = in_channels
        self.out_channel: list[int] = out_channel
        self.out_channels: int = out_channel
        self.out_channels = in_channels[1:] + [out_channel]

        self.kernel_generator = KernelGeneratorMLP(
            input_metadata_dim,
            hidden_dims,
            self.in_channels,
            self.out_channels,
            kernel_shapes
        )

        self.conv_net = DynamicConvNet(
            self.in_channels,
            self.out_channels,
            kernel_shapes
        )

    def forward(self,
                x_meatadata: torch.Tensor,
                x: torch.Tensor
                ) -> torch.Tensor:
        kernels = self.kernel_generator(x_meatadata)
        batch_size: int = x.shape[0]
        num_channels: int = x.shape[1]
        x = x.view(1, batch_size * num_channels, x.shape[2], x.shape[3])
        x = self.conv_net(x, kernels, batch_size)
        return x.view(batch_size, self.out_channel, x.shape[2], x.shape[3])


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
