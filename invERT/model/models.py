import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import zip_longest
import numpy as np




class UNet_basic(nn.Module):
    def __init__(self):
        super(UNet_basic, self).__init__()

        self.batch_processing = False

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=16, groups=1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, groups=16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mlp = nn.Sequential(
            nn.Linear(32 * 4 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 32 * 4 * 8),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
        )
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
        )
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
        )
        self.upconv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)


    def forward(self, x, target):
        x = self.conv_block1(x)
        if x.shape[-2] >= 8:
            x = self.maxpool(x)
        x = self.conv_block2(x)
        if x.shape[-2] >= 8:
            x = self.maxpool(x)
        
        if x.shape[-2:] != (4, 8):
            x = F.interpolate(x, size=(4, 8), mode='bilinear', align_corners=True)
        
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = x.view(x.size(0), 32, 4, 8)
        x = self.upconv1(x)
        x = self.upconv2(x)
        if x.shape[-2] < target.shape[-2]:
            x = self.upconv3(x)
        if x.shape[-2] < target.shape[-2]:
            x = self.upconv4(x)
        if x.shape[-2] < target.shape[-2]:
            x = self.upconv5(x)

        if x.shape[-2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)

        # Final output layer maps to desired number of channels.
        return F.sigmoid(self.final_conv(x))










class DoubleConv(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels, kernel_size, stride, padding, dilation, bias=False),
            nn.InstanceNorm2d(num_out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(num_out_channels, num_out_channels, kernel_size, stride, padding, dilation, bias=False),
            nn.InstanceNorm2d(num_out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpSample(nn.Module):
    def __init__(self, feature):
        super(UpSample, self).__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # Upsample layer
            nn.Conv2d(feature * 2, feature, kernel_size=1), # e.g., 1x1 conv
            nn.InstanceNorm2d(feature),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
    
    def forward(self, x):
        return self.upsample(x)


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
        self.batch_processing = False

        num_input_channels = 5
        features = [8, 16, 32, 64, 128]
        kernel_sizes = [3, 3, 3, 3, 3]
        strides = [1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1]
        dilations = [1, 1, 1, 1, 1]

        bottleneck_in, bottleneck_out = 128, 256
        # Downsampling path (Encoder)
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Define pooling layer

        for feature, kernel_size, stride, padding, dilation in zip(features, kernel_sizes, strides, paddings, dilations):
            self.downs.append(DoubleConv(num_input_channels, feature, kernel_size, stride, padding, dilation))
            num_input_channels = feature
        
        # Bottleneck layer (bottom of U-Net)
        self.bottleneck = DoubleConv(bottleneck_in, bottleneck_out, kernel_size=3, stride=1, padding=1, dilation=1)
        
        # Upsampling path (Decoder)
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                UpSample(feature)  # Upsample layer
            )
            # Double conv after concatenating with corresponding feature map from encoder
            self.ups.append(
                DoubleConv(feature * 2, feature, kernel_size=3, stride=1, padding=1, dilation=1)
            )

        # Final output layer maps to desired number of channels.
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(features[0], 1, kernel_size=1)
        )
    
    def forward(self, x, target):
        skip_connections = []

        # Downsampling: store outputs for skip connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            if x.shape[-2] > 8:  # Check spatial dimension before pooling
                x = self.pool(x)  # Apply max pooling after each downsampling block
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse for upward path
        # Upsampling: use skip connections for precise localization
        idx_skip = 0
        for idx in range(0, len(self.ups), 2):
            # Upsample
            x = self.ups[idx](x)
            # Concatenate with corresponding skip connection
            skip_connection = skip_connections[idx_skip]
            idx_skip += 1

            # Handle potential mismatches in dimensions (due to odd input dimensions)
            if x.shape != skip_connection.shape:
                skip_connection = F.interpolate(skip_connection, size=x.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate along the channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # Apply the double convolution after concatenation
            x = self.ups[idx + 1](concat_skip)

        x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        
        return F.sigmoid(self.final_conv(x))












class DoubleConv_1(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, kernel_size=3, stride=1, padding=1):
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


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, use_transpose_conv=True):
        super().__init__()

        if use_transpose_conv:
            # Use ConvTranspose2d: learns the upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # Input to conv block: channels from skip connection + channels from up-convolution
            self.conv = DoubleConv(out_channels + in_channels // 2, out_channels)
        else:
            # Use Upsample + Conv: fixed bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Input to conv block: channels from skip connection + channels from upsampled features (which is in_channels)
            self.conv = DoubleConv(out_channels + in_channels, out_channels)


    def forward(self, x_to_upsample, x_skip):
        x_up = self.up(x_to_upsample)

        # Interpolate skip connection to match upsampled size if needed (robust way)
        if x_skip.shape[2:] != x_up.shape[2:]:
            x_skip = F.interpolate(x_skip, size=x_up.shape[2:], mode='bilinear', align_corners=True)

        x_concat = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x_concat)


class UNet_1(nn.Module):
    """
    U-Net architecture for image-to-image translation.
    
    Args:
        input_channels (int): Number of channels in the input image.
        output_channels (int): Number of channels in the output image.
        features (list of int): Number of features at each level in the encoder.
    """
    def __init__(self):
        super(UNet, self).__init__()

        num_input_channels = 5
        features = [8, 16, 32, 64, 128]
        kernel_sizes = [3, 3, 3, 3, 3]
        strides = [1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1]

        bottleneck_in, bottleneck_out = 128, 256
        # Downsampling path (Encoder)
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Define pooling layer

        for feature, kernel_size, stride, padding in zip(features, kernel_sizes, strides, paddings):
            self.downs.append(DoubleConv(num_input_channels, feature, kernel_size, stride, padding))
            num_input_channels = feature
        
        # Bottleneck layer (bottom of U-Net)
        self.bottleneck = DoubleConv(bottleneck_in, bottleneck_out, kernel_size=3, stride=1, padding=1)
        
        # Upsampling path (Decoder)
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                Up(feature * 2, feature, use_transpose_conv=True)  # Upsample layer
            )

        # Final output layer maps to desired number of channels.
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)
    
    def forward(self, x, target):
        skip_connections = []

        # Downsampling: store outputs for skip connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            if x.shape[-2] != 1:
                x = self.pool(x)  # Apply max pooling after each downsampling block
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse for upward path

        for idx in range(len(self.ups)):
            # Upsample
            x = self.ups[idx](x)

        x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
        
        return F.sigmoid(self.final_conv(x))






























class AttentionGate(nn.Module):
    """
    Attention Gate mechanism based on "Attention U-Net".
    Filters features passed through skip connections.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of channels in the gating signal (from deeper decoder layer).
            F_l: Number of channels in the skip connection (from encoder).
            F_int: Number of channels in the intermediate layer.
        """
        super(AttentionGate, self).__init__()

        # Convolution for gating signal (reduce channels)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
            # No ReLU here as per original paper diagram (applied after addition)
        )

        # Convolution for skip connection (reduce channels)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Final convolution to get attention coefficients (1 channel)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from deeper layer (needs upsampling potentially).
            x: Skip connection features from encoder.

        Returns:
            Attended skip connection features (same shape as x).
        """
        # Process gating signal
        g1 = self.W_g(g) # Shape: [B, F_int, H_g, W_g]

        # Process skip connection
        x1 = self.W_x(x) # Shape: [B, F_int, H_x, W_x]

        # Add processed signals. Upsample g1 if needed to match x1's spatial dims.
        # Typically, g comes from a layer with half the spatial resolution of x.
        # We can upsample g1 before adding.
        if g1.shape != x1.shape:
            # Using align_corners=True consistent with potential interpolation elsewhere
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        # Combine signals and apply ReLU
        psi_input = self.relu(g1 + x1)

        # Generate attention map (alpha)
        alpha = self.psi(psi_input) # Shape: [B, 1, H_x, W_x]

        # Apply attention map to original skip connection (element-wise multiplication)
        return x * alpha


class UNetAttention(nn.Module):
    """
    U-Net architecture modified with Attention Gates in the skip connections.
    """
    def __init__(self):
        super(UNetAttention, self).__init__()

        num_input_channels = 5
        features = [32, 64, 128] # F_l values for attention gates
        kernel_sizes = [7, 3, 3]
        strides = [1, 1, 1]
        paddings = [3, 1, 1]

        self.downs = nn.ModuleList()
        self.att_gates = nn.ModuleList() # Store attention gates
        self.ups = nn.ModuleList()

        # --- Encoder ---
        current_channels = num_input_channels
        for feature, kernel_size, stride, padding in zip(features, kernel_sizes, strides, paddings):
            self.downs.append(DoubleConv(current_channels, feature, kernel_size, stride, padding))
            current_channels = feature

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, kernel_size=3, stride=1, padding=1)
        current_channels = features[-1] * 2 # Channels out of bottleneck

        # --- Decoder ---
        for feature in reversed(features):
            # Gating signal comes from the output of the layer below (current_channels)
            F_g = current_channels
            # Skip connection comes from encoder layer with 'feature' channels
            F_l = feature
            # Intermediate channels for attention (can be tuned, e.g., F_l // 2)
            F_int = feature // 2

            # Add Attention Gate: Takes Gating signal (F_g) and Skip connection (F_l)
            self.att_gates.append(AttentionGate(F_g=F_g, F_l=F_l, F_int=F_int))

            # Add Upsampling ConvTranspose
            self.ups.append(
                nn.ConvTranspose2d(current_channels, feature, kernel_size=2, stride=2)
            )
            current_channels = feature # Channels after upsampling

            # Add DoubleConv. Input channels = channels from upsample + channels from (attended) skip connection
            # The attended skip connection (output of AttentionGate) has F_l = 'feature' channels.
            self.ups.append(
                DoubleConv(feature * 2, feature, kernel_size=3, stride=1, padding=1)
            )
            current_channels = feature # Channels after DoubleConv

        # --- Final Layer ---
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, target):
        skip_connections_encoder = []

        # --- Downsampling / Encoder ---
        current_features = x
        for down_block in self.downs:
            current_features = down_block(current_features)
            skip_connections_encoder.append(current_features)
            # Apply pooling if spatial dimensions are large enough
            if current_features.shape[2] > 4: # Check spatial dimension before pooling
                 current_features = self.pool(current_features)

        # --- Bottleneck ---
        bottleneck_features = self.bottleneck(current_features)

        # Reverse skip connections for decoder path
        skip_connections_encoder = skip_connections_encoder[::-1]
        # Reverse attention gates to match decoder path
        # attention_gates = self.att_gates[::-1]

        # --- Upsampling / Decoder ---
        current_features = bottleneck_features
        # Iterate through the decoder stages (upsampling + double conv pairs)
        for i in range(0, len(self.ups), 2):
            # Index for skip connections and attention gates (0, 1, 2)
            decoder_level_idx = i // 2

            # Get the corresponding attention gate (DO NOT REVERSE self.att_gates)
            # self.att_gates is already ordered [att_for_128, att_for_64, att_for_32]
            att_gate = self.att_gates[decoder_level_idx]

            # Gating signal is the features from the deeper layer
            gating_signal = current_features
            # Skip connection is from the corresponding encoder layer
            skip_connection = skip_connections_encoder[decoder_level_idx]

            # Apply attention gate
            attended_skip = att_gate(g=gating_signal, x=skip_connection)

            # Upsample current features (ConvTranspose2d)
            current_features = self.ups[i](current_features)

            # Handle potential dimension mismatch after ConvTranspose
            if current_features.shape[2:] != attended_skip.shape[2:]:
                 current_features = F.interpolate(current_features, size=attended_skip.shape[2:], mode='bilinear', align_corners=True)

            # Concatenate attended skip connection and upsampled features
            concat_features = torch.cat((attended_skip, current_features), dim=1)

            # Apply DoubleConv block
            current_features = self.ups[i + 1](concat_features)

        # --- Final Resizing and Output Conv ---
        # Use target shape for final interpolation - THIS REMAINS THE SAME
        final_features = F.interpolate(current_features, size=target.shape[2:], mode='bilinear', align_corners=True)

        return self.final_conv(final_features)



def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    pass
