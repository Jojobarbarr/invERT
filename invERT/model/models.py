import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import zip_longest
import numpy as np


class Layer:
    def __init__(self,
                 layer_type: str,
                 kernel_shape: tuple[int] | int,
                 stride: tuple[int] | int,
                 padding: tuple[int] | int,
                 dilation: int,
                 num_in_channels: int,
                 ) -> None:
        self.type = layer_type
        self.layer_dim = 1 if "1D" in layer_type else 2
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.num_in_channels = num_in_channels
        self.num_weights = 0
    
    def __str__(self):
        return (
            f"{self.type=}, {self.layer_dim=}, "
            f"{self.kernel_shape=}, {self.stride=}, "
            f"{self.padding=}, {self.dilation=}, "
            f"{self.num_in_channels=}"
        )
    


class KernelGeneratorMLP(nn.Module):
    def __init__(self,
                 input_metadata_dim: int,
                 hidden_dims: list[int],
                 num_in_channels: list[int | list[list[int]]],
                 num_out_channels: list[int | list[list[int]]],
                 cnn_layers: list[Layer | list[list[Layer]]],
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
        self.num_in_channels: list[int | list[list[int]]] = num_in_channels
        self.num_out_channels: list[int | list[list[int]]] = num_out_channels
        self.num_conv_layers: int = len(num_in_channels)
        self.conv_layers: list[Layer] = []

        # Compute the number of weights needed for each conv layer
        self.num_weights: list[int] = []
        
        for layer_idx, layer in enumerate(cnn_layers):
            if isinstance(layer, list):
                if len(layer[1]) > 0:
                    for sub_idx, (sub_layer_0, sub_layer_1) in enumerate(zip_longest(layer[0], layer[1])):
                        max_num_weights, idx = torch.max(
                            torch.tensor([
                                self._compute_num_weights(layer_idx, sub_layer_0, 0, sub_idx),
                                self._compute_num_weights(layer_idx, sub_layer_1, 1, sub_idx)
                            ]),
                            0
                        )
                        self.num_weights.append(max_num_weights)
                        self.conv_layers.append(sub_layer_0 if idx == 0 else sub_layer_1)
                else:
                    for sub_idx, sub_layer_0 in enumerate(layer[0]):
                        self.num_weights.append(self._compute_num_weights(layer_idx, sub_layer_0, 0, sub_idx))
                        self.conv_layers.append(sub_layer_0)
            else:
                self.num_weights.append(
                    self._compute_num_weights(layer_idx, layer)
                )
                self.conv_layers.append(layer)

        self.total_nbr_weights: int = sum(self.num_weights)
        print(f"Max total number of weights for CNN: {self.total_nbr_weights}")

        # Build the MLP as a sequential module.
        layers = []
        layers.append(nn.Linear(input_metadata_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], self.total_nbr_weights))
        self.mlp = nn.Sequential(*layers)

    def _compute_num_weights(self,
                             layer_idx: int,
                             layer: Layer,
                             path_idx: int = 0,
                             sub_idx: int = 0
                             ) -> int:
        if layer is None:
            return 0
        if layer.type == "conv1D":
            # 47 comes from the maximum pseudodepth a sample can have.
            return 47 * layer.kernel_shape
        if layer.type == "conv2D":
            # if isinstance(self.num_in_channels, list):
            #     return (
            #         self.num_in_channels[layer_idx][path_idx][sub_idx]
            #         * self.num_out_channels[layer_idx][path_idx][sub_idx]
            #         * layer.kernel_shape[0]
            #         * layer.kernel_shape[1]
            #     )            
            # return (
            #     self.num_in_channels[layer_idx]
            #     * self.num_out_channels[layer_idx]
            #     * layer.kernel_shape[0]
            #     * layer.kernel_shape[1]
            # )
            # TODO: handle channels
            return layer.kernel_shape[0] * layer.kernel_shape[1]
        if layer.type == "convT2D":
            # if isinstance(self.num_in_channels, list):
            #     return (
            #         self.num_in_channels[layer_idx][path_idx][sub_idx]
            #         * self.num_out_channels[layer_idx][path_idx][sub_idx]
            #         * layer.kernel_shape[0]
            #         * layer.kernel_shape[1]
            #     )
            # return (
            #     self.num_in_channels[layer_idx]
            #     * self.num_out_channels[layer_idx]
            #     * layer.kernel_shape[0]
            #     * layer.kernel_shape[1]
            # )
            return layer.kernel_shape[0] * layer.kernel_shape[1]
        return 0


    def forward(self,
                x: torch.Tensor,
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
        for layer_idx, layer in enumerate(self.conv_layers):
            end_idx = start_idx + self.num_weights[layer_idx]
            if layer.type == "conv1D":
                kernel = x[:, start_idx:end_idx].reshape(
                    47,
                    1, # 47 comes from the maximum pseudodepth a sample can have.
                    layer.kernel_shape
                )
            elif layer.type == "conv2D":
                # kernel = x[:, start_idx:end_idx].reshape(
                #     self.num_in_channels[layer_idx][0],
                #     self.num_out_channels[layer_idx][0],
                #     layer.kernel_shape[0],
                #     layer.kernel_shape[1]
                # )
                kernel = x[:, start_idx:end_idx].reshape(
                    1,
                    1,
                    layer.kernel_shape[0],
                    layer.kernel_shape[1]
                )
            elif layer.type == "convT2D":
                # kernel = x[:, start_idx:end_idx].reshape(
                #     self.num_in_channels[layer],
                #     self.num_out_channels[layer],
                #     layer.kernel_shape[0],
                #     layer.kernel_shape[1]
                # )
                kernel = x[:, start_idx:end_idx].reshape(
                    1,
                    1,
                    layer.kernel_shape[0],
                    layer.kernel_shape[1]
                )
            output_list.append(kernel)
            start_idx = end_idx
        return output_list


class DynamicConv2D(nn.Module):
    def __init__(self,
                 stride: tuple[int],
                 padding: tuple[int],
                 dilation: int
                 ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(
            x,
            kernels,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        x = self.batch_norm(x)
        return x

class DynamicConv1D(nn.Module):
    def __init__(self,
                 stride: tuple[int],
                 padding: tuple[int],
                 dilation: int
                 ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        x = x.view(1, x.shape[2], x.shape[3])
        x = F.conv1d(
            x,
            kernels[:x.shape[2], :],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=x.shape[1],
        )
        x = self.batch_norm(x.view(1, 1, x.shape[1], x.shape[2]))
        return x

class DynamicConvTranspose2D(nn.Module):
    def __init__(self, stride: tuple[int], padding: tuple[int], dilation: int) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        x = F.conv_transpose2d(
            x,
            kernels,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        x = self.batch_norm(x)
        return x


class DynamicCNN(nn.Module):
    def __init__(self,
                 cnn_layers: list[Layer | list[list[Layer]]],
                 num_in_channel: list[int | list[list[int]]],
                 num_out_channel: list[int | list[list[int]]],
                 threshold: int,
                 ) -> None:
        super().__init__()

        self.layers_path_1 = nn.ModuleList()
        self.layers_caract_1: list[Layer] = []
        self.num_in_channel_1: list[int] = []
        self.num_out_channel_1: list[int] = []
        self.num_weights_1: list[int] = []
        self.layers_path_2 = nn.ModuleList()
        self.layers_caract_2: list[Layer] = []
        self.num_in_channel_2: list[int] = []
        self.num_out_channel_2: list[int] = []
        self.num_weights_2: list[int] = []

        self.threshold = threshold

        for layer_idx, layer in enumerate(cnn_layers):
            if isinstance(layer, list):
                for sub_idx, layer_path_1 in enumerate(layer[0]):
                    if len(self.num_in_channel_1) > 0:
                        self.num_out_channel_1.append(self.num_in_channel_1[-1])
                    self.num_in_channel_1.append(num_in_channel[layer_idx][0][sub_idx])
                    self.layers_path_1.append(self._parse_layer(layer_path_1))
                    self.layers_caract_1.append(layer_path_1)

                for sub_idx, layer_path_2 in enumerate(layer[1]):
                    if len(self.num_in_channel_2) > 0:
                        self.num_out_channel_2.append(self.num_in_channel_2[-1])
                    self.num_in_channel_2.append(num_in_channel[layer_idx][1][sub_idx])
                    self.layers_path_2.append(self._parse_layer(layer_path_2))
                    self.layers_caract_2.append(layer_path_2)
            else:
                lay = self._parse_layer(layer)
                if len(self.num_in_channel_1) > 0:
                    self.num_out_channel_1.append(self.num_in_channel_1[-1])
                if len(self.num_in_channel_2) > 0:
                    self.num_out_channel_2.append(self.num_in_channel_2[-1])
                self.layers_path_1.append(lay)
                self.layers_caract_1.append(layer)
                self.num_in_channel_1.append(num_in_channel[layer_idx])
                self.layers_path_2.append(lay)
                self.layers_caract_2.append(layer)
                self.num_in_channel_2.append(num_in_channel[layer_idx])
            
        self.num_out_channel_1.append(num_out_channel[-1])
        self.num_out_channel_2.append(num_out_channel[-1])
        

        for num_weights, layers_caract, num_in_channels, num_out_channels in zip(
            [self.num_weights_1, self.num_weights_2],
            [self.layers_caract_1, self.layers_caract_2],
            [self.num_in_channel_1, self.num_in_channel_2],
            [self.num_out_channel_1, self.num_out_channel_2],
        ):
            for layer_idx, layer in enumerate(layers_caract):
                if layer.type == "conv1D":
                    num_weights.append(0)  # Determined at forward time
                if layer.type == "conv2D":
                    num_weights.append(
                        num_in_channels[layer_idx]
                        * num_out_channels[layer_idx]
                        * layer.kernel_shape[0]
                        * layer.kernel_shape[1]
                    )
                if layer.type == "convT2D":
                    num_weights.append(
                        num_in_channels[layer_idx]
                        * num_out_channels[layer_idx]
                        * layer.kernel_shape[0]
                        * layer.kernel_shape[1]
                    )

      
    def _parse_layer(self, layer: Layer) -> nn.Module:
        if layer.type == "conv2D":
            lay = DynamicConv2D(layer.stride, layer.padding, layer.dilation)
        elif layer.type == "conv1D":
            lay = DynamicConv1D(layer.stride, layer.padding, layer.dilation)
        elif layer.type == "convT2D":
            lay = DynamicConvTranspose2D(layer.stride, layer.padding, layer.dilation)
        else:
            raise ValueError(f"Unknown layer type: {layer.type}")
        return lay
    

    def forward(self,
                x: torch.Tensor,
                kernels: list[torch.Tensor],
                target: torch.Tensor
                ) -> torch.Tensor:

        if target.shape[-1] >= self.threshold:
            layers = self.layers_path_1
            layers_caract = self.layers_caract_1
            num_weights = self.num_weights_1
        else:
            layers = self.layers_path_2
            layers_caract = self.layers_caract_2
            num_weights = self.num_weights_2
        
        for layer_idx, (layer, layer_caract, num_weight) in enumerate(zip(layers, layers_caract, num_weights)):
            if layer_idx == len(layers) - 1:
                x = F.interpolate(x, size=(target.shape[2], target.shape[3]), mode="bilinear", align_corners=False)
                break
            if layer_caract.type == "conv1D":
                x = layer(x, kernels[layer_idx][:x.shape[2]])
                x = F.relu(x)
            elif layer_caract.type == "conv2D" or layer_caract.type == "convT2D":
                x = layer(x, kernels[layer_idx].view(-1)[:num_weight].reshape(1, 1, layer_caract.kernel_shape[0], layer_caract.kernel_shape[1]))
                x = F.relu(x)
        
        x = layers[-1](x, kernels[-1].view(-1)[:num_weights[-1]].reshape(1, 1, layer_caract.kernel_shape[0], layer_caract.kernel_shape[1]))
        return x


class DynamicModel(nn.Module):
    def __init__(self,
                 input_metadata_dim: int,
                 hidden_dims: list[int],
                 cnn_layers: list[Layer | list[list[Layer]]],
                 num_out_channels: int,
                 threshold: int,
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
        self.conv_num_in_channels: list[int | list[list[int]]] = []
        for layer in cnn_layers:
            if isinstance(layer, list):
                sub_layer_in_channels: list[list[int]] = [[], []]
                for sub_layer in layer[0]:
                    sub_layer_in_channels[0].append(sub_layer.num_in_channels)
                if len(layer) > 1:
                    for sub_layer in layer[1]:
                        sub_layer_in_channels[1].append(sub_layer.num_in_channels)
                self.conv_num_in_channels.append(sub_layer_in_channels)
            else:
                self.conv_num_in_channels.append(layer.num_in_channels)
        
        self.conv_num_out_channels: list[int | list[list[int]]] = \
            self.conv_num_in_channels[1:] + [num_out_channels]

        self.conv_net = DynamicCNN(
            cnn_layers,
            self.conv_num_in_channels,
            self.conv_num_out_channels,
            threshold,
        )

        self.kernel_generator = KernelGeneratorMLP(
            input_metadata_dim,
            hidden_dims,
            self.conv_num_in_channels,
            self.conv_num_out_channels,
            cnn_layers,
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
    
# Attention Block: Gate mechanism to weight encoder features before fusion.
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: Number of channels in the gating signal (from decoder).
        F_l: Number of channels in the encoder features.
        F_int: Number of intermediate channels.
        """
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, g):
        """
        x: Encoder feature map (skip connection)
        g: Decoder feature map (gating signal)
        """
        # Apply the 1x1 convolutions
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Combine and apply ReLU
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Multiply attention mask to the encoder feature map
        return x * psi

# Down-sampling block: maxpool followed by double conv.
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

# Up-sampling block: upsample + attention gate before concatenating with the skip connection.
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # If bilinear, use the normal convolutions to reduce the number of channels.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # If not bilinear, learn transposed convolution.
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
        # Attention block: Note that g (decoder) uses lower channels.
        # Here we assume the skip connection has out_channels channels
        self.attention = AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
    
    def forward(self, x1, x2):
        # x1: feature map from decoder, x2: corresponding encoder feature map (skip connection)
        x1 = self.up(x1)
        
        # Ensure x1 is the same size as x2 (if input dimensions may be off by one)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Apply attention gate on skip connection
        x2 = self.attention(x2, x1)
        
        # Concatenate along the channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

# Full Attention U-Net architecture combining all the blocks.
class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        n_channels: number of input channels (e.g., 1 for grayscale, 3 for RGB)
        n_classes: number of output classes (for segmentation tasks)
        """
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1  # adjust the number of channels if bilinear is used
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x):
        x1 = self.inc(x)       # First convolution block
        x2 = self.down1(x1)    # Downsample 1
        x3 = self.down2(x2)    # Downsample 2
        x4 = self.down3(x3)    # Downsample 3
        x5 = self.down4(x4)    # Bottom layer
        
        # Up-sampling path with attention applied on skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, kernel_size, stride, padding):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(num_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_out_channels, num_out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(num_out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    U-Net architecture for image-to-image translation.
    
    Args:
        input_channels (int): Number of channels in the input image.
        output_channels (int): Number of channels in the output image.
        features (list of int): Number of features at each level in the encoder.
    """
    def __init__(self):
        super(UNet, self).__init__()

        num_input_channels = 1
        features = [64, 128, 256]
        kernel_sizes = [7, 3, 3]
        strides = [1, 1, 1]
        paddings = [3, 1, 1]
        # Downsampling path (Encoder)
        self.downs = nn.ModuleList()
        for feature, kernel_size, stride, padding in zip(features, kernel_sizes, strides, paddings):
            self.downs.append(DoubleConv(num_input_channels, feature, kernel_size, stride, padding))
            num_input_channels = feature
        
        # Bottleneck layer (bottom of U-Net)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, kernel_size=3, stride=1, padding=1)
        
        # Upsampling path (Decoder)
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            # Transposed convolution for upsampling
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # Double conv after concatenating with corresponding feature map from encoder
            self.ups.append(
                DoubleConv(feature * 2, feature, kernel_size=3, stride=1, padding=1)
            )

        # Final output layer maps to desired number of channels.
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, target):
        skip_connections = []

        # Downsampling: store outputs for skip connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            if x.shape[2] > 4:
                x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse for upward path
        
        # Upsampling: use skip connections for precise localization
        for idx in range(0, len(self.ups), 2):
            # Upsample
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            # Handle potential mismatches in dimensions (due to odd input dimensions)
            if x.shape != skip_connection.shape:
                skip_connection = F.interpolate(skip_connection, size=x.shape[2:], mode='bilinear', align_corners=True)
            # Concatenate along the channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # Apply the double convolution after concatenation
            x = self.ups[idx + 1](concat_skip)

        x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
        
        return self.final_conv(x)


class MLP_huge(nn.Module):
    def __init__(self):
        super(MLP_huge, self).__init__()
        self.fc1 = nn.Linear(2209, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 4000)
    
    def forward(self, x, target):
        if x.shape[0] != 2209:
            x = F.pad(x, (0, 2209 - x.shape[0], "constant", 0))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc4(x)
        x = x.view(200, 200)
        x = F.interpolate(x, size=target.shape, mode='bilinear', align_corners=False)
        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    pass
