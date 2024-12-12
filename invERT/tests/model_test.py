import unittest
from invERT.model.models import DynamicModel, KernelGeneratorMLP, DynamicConvNet
import torch


class TestKernelGeneratorMLP(unittest.TestCase):
    def test_forward(self):
        input_dim: int = 2
        hidden_dim: list[int] = [64, 128]
        nbr_kernel: list[int] = [16, 1]
        kernel_sizes: list[int] = [3, 3]

        nbr_weight_layers = [nbr_kernel[i] * kernel_sizes[i]
                             ** 2 for i in range(len(nbr_kernel))]
        output_layer_size = sum(nbr_weight_layers)
        print(output_layer_size)

        model = KernelGeneratorMLP(input_dim,
                                   hidden_dim,
                                   nbr_kernel,
                                   kernel_sizes
                                   )

        input_tensor = torch.rand((16, 2))
