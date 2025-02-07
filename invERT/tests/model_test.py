import unittest
from invERT.model.models import KernelGeneratorMLP, DynamicConv2D, \
    DynamicConvNet, DynamicModel
import torch


class TestKernelGeneratorMLP(unittest.TestCase):
    def test___init__(self):
        input_metadata_dim: int = 8
        hidden_dims: list[int] = [16, 64]
        in_channels: list[int] = [1, 32]
        out_channels: list[int] = [16, 2]
        kernel_shapes: list[int] = [3, 5]
        KG_model: KernelGeneratorMLP = KernelGeneratorMLP(
            input_metadata_dim,
            hidden_dims,
            in_channels,
            out_channels,
            kernel_shapes
        )
        nbr_weights_conv_layers: list[int] = [1 * 16 * 3 * 3, 32 * 2 * 5 * 5]
        self.assertListEqual(KG_model.nbr_weights_conv_layers,
                             nbr_weights_conv_layers)
        # One layer out of two is a ReLU layer
        self.assertEqual(KG_model.mlp[0].in_features, 8)
        self.assertEqual(KG_model.mlp[0].out_features, 16)
        self.assertEqual(KG_model.mlp[2].in_features, 16)
        self.assertEqual(KG_model.mlp[2].out_features, 64)
        self.assertEqual(KG_model.mlp[4].in_features, 64)
        self.assertEqual(KG_model.mlp[4].out_features,
                         sum(nbr_weights_conv_layers))

    def test_forward(self):
        input_metadata_dim: int = 8
        hidden_dims: list[int] = [16, 64]
        in_channels: list[int] = [4, 32]
        out_channels: list[int] = [16, 2]
        kernel_shapes: list[int] = [3, 5]
        KG_model: KernelGeneratorMLP = KernelGeneratorMLP(
            input_metadata_dim,
            hidden_dims,
            in_channels,
            out_channels,
            kernel_shapes
        )
        # input_tensor is of size (batch_size, input_dim)
        input_tensor: torch.Tensor = torch.randn(16, 8)
        output_tensor: torch.Tensor = KG_model(input_tensor)
        self.assertEqual(len(output_tensor), 2)

        self.assertEqual(output_tensor[0].shape[0], 16 * 16)
        self.assertEqual(output_tensor[0].shape[1], 4)
        self.assertEqual(output_tensor[0].shape[2], 3)
        self.assertEqual(output_tensor[0].shape[3], 3)

        self.assertEqual(output_tensor[1].shape[0], 16 * 2)
        self.assertEqual(output_tensor[1].shape[1], 32)
        self.assertEqual(output_tensor[1].shape[2], 5)
        self.assertEqual(output_tensor[1].shape[3], 5)


class TestDynamicConv2D(unittest.TestCase):
    def test_forward(self):
        stride: int = 1
        padding: str = "same"

        DC2D_model: DynamicConv2D = DynamicConv2D(
            stride,
            padding
        )
        # kernels are of size
        # (out_channels, in_channels // batch_size, kernel_shape, kernel_shape)
        kernels = torch.randn(64, 8, 3, 3)
        batch_size: int = 4
        # input_tensor is of size
        # (batch_size, in_channels, input_dim, input_dim)
        input_tensor: torch.Tensor = torch.randn(4, 32, 256, 256)
        output_tensor: torch.Tensor = DC2D_model(
            input_tensor, kernels, batch_size)
        self.assertEqual(output_tensor.shape[0], 4)
        self.assertEqual(output_tensor.shape[1], 64)
        self.assertEqual(output_tensor.shape[2], 256)
        self.assertEqual(output_tensor.shape[3], 256)


class TestDynamicConvNet(unittest.TestCase):
    def test_forward(self):
        in_channels: list[int] = [32, 64]
        DCmodel: DynamicConvNet = DynamicConvNet(in_channels)

        input_tensor: torch.Tensor = torch.randn(4, 32, 256, 256)
        kernels = [torch.randn(64, 8, 3, 3), torch.randn(64, 16, 3, 3)]
        batch_size: int = 4
        output_tensor: torch.Tensor = DCmodel(
            input_tensor, kernels, batch_size)
        self.assertEqual(output_tensor.shape[0], 4)
        self.assertEqual(output_tensor.shape[1], 64)
        self.assertEqual(output_tensor.shape[2], 256)
        self.assertEqual(output_tensor.shape[3], 256)


class TestDynamicModel(unittest.TestCase):
    def test_forward(self):
        input_metadata_dim: int = 8
        hidden_dims: list[int] = [16, 64]
        in_channels: list[int] = [32, 64]
        out_channel: int = 2
        kernel_shapes: list[int] = [3, 5]
        DM_model: DynamicModel = DynamicModel(
            input_metadata_dim,
            hidden_dims,
            in_channels,
            out_channel,
            kernel_shapes
        )
        # input_tensor is of size
        # (batch_size, in_channels, input_dim, input_dim)
        input_tensor: torch.Tensor = torch.randn(4, 32, 256, 256)
        # input_metadata is of size (batch_size, input_metadata_dim)
        input_metadata: torch.Tensor = torch.randn(4, 8)
        output_tensor: torch.Tensor = DM_model(input_metadata, input_tensor)
        self.assertEqual(output_tensor.shape[0], 4)
        self.assertEqual(output_tensor.shape[1], 2)
        self.assertEqual(output_tensor.shape[2], 256)
        self.assertEqual(output_tensor.shape[3], 256)
