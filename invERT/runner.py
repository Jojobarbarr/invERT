from torch import device as set_device
import torch
from torch import Tensor
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available
from torch.optim import Adam, SGD, RMSprop
from torch.optim.optimizer import Optimizer
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss, L1Loss
from config.configuration import Config
from model.models import DynamicModel, Layer, UNet, MLP_huge
from pathlib import Path
from model.train import train
from torch.utils.data import random_split
from data.data import (
    LMDBDataset,
    InvERTSample
)
from typing import Type
import matplotlib.pyplot as plt
import multiprocessing as mp
from model.parameters_classes import LoggingParameters



def go_through_cnn_layers(cnn_config: Config, cnn_layers: list):
    layers_list: list[Layer | list[Layer]] = []
    for layer in cnn_layers:
        if layer.layer_type == "branch":
            layers_first_choice = go_through_cnn_layers(cnn_config, layer.options[0])
            if len(layer.options) > 1:
                layers_second_choice = go_through_cnn_layers(cnn_config, layer.options[1])
            layers_list.append(
                [
                    layers_first_choice,
                    layers_second_choice
                ]
            )
        else:
            if "1D" in layer.layer_type:
                kernel_shape: int = layer.kernel_width
                stride: int = layer.stride_width
                padding: int = layer.padding_width
                num_in_channels: int = 0
            else:
                kernel_shape: tuple[int] = (layer.kernel_width, layer.kernel_height)
                stride: tuple[int] = (layer.stride_width, layer.stride_height)
                padding: tuple[int] = (layer.padding_width, layer.padding_height)
                num_in_channels: int = layer.num_in_channels
            dilation: int = layer.dilation
            layers_list.append(
                Layer(
                    layer_type=layer.layer_type,
                    kernel_shape=kernel_shape,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    num_in_channels=num_in_channels,
                )
            )
    return layers_list

def init_model(config: Config) -> DynamicModel:
    # mlp_config: Config = config.model.mlp
    # cnn_config: Config = config.model.cnn

    # layers_list: list[Layer | list[list[Layer]]] = go_through_cnn_layers(cnn_config, cnn_config.cnn_layers)

    # num_out_channels: int = cnn_config.num_out_channels

    # threshold: int = 30

    # model = DynamicModel(
    #     mlp_config.input_metadata_dim,
    #     mlp_config.hidden_dims,
    #     layers_list,
    #     num_out_channels,
    #     threshold,
    # )

    # model = UNet()

    model = MLP_huge()

    return model


def init_optimizer(config: Config,
                   model: DynamicModel
                   ) -> Optimizer:
    initial_lr: float = config.training.initial_learning_rate

    optimizer_config: Config = config.training.optimizer

    optimizer: Optimizer
    if optimizer_config.type == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=initial_lr,
        )
    elif optimizer_config.type == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=initial_lr,
        )
    elif optimizer_config.type == "rmsprop":
        optimizer = RMSprop(
            model.parameters(),
            lr=initial_lr,
        )

    return optimizer


def init_scheduler(config: Config,
                   optimizer
                   ) -> Module:
    scheduler_config: Config = config.training.lr_scheduler

    scheduler: Module
    if scheduler_config.enabled:
        lr_scheduler_type = scheduler_config.type
        if lr_scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.factor,
                patience=scheduler_config.patience)
    else:
        scheduler = None

    return scheduler


def init_logging(config: Config,
                 train_dataloader: DataLoader
                 ) -> tuple[set[int], int]:
    logging_config: Config = config.logging
    num_print_points: int = logging_config.num_print_points
    num_batches: int = len(train_dataloader)
    step_between_print: int = num_batches // num_print_points
    print_points: set[int] = {
        i * step_between_print + (epoch * num_batches)
        for i in range(1, num_print_points)
        for epoch in range(config.training.epochs)
    }

    return print_points



class Transform:
    def __init__(self):
        pass

    def __call__(self, sample: InvERTSample) -> InvERTSample:
        sample['pseudosection'] = self.pad(sample['pseudosection'])
        sample = {
            key: value.unsqueeze(0).unsqueeze(0)
            if key != "pseudosection" and key != "norm_log_resistivity_model" else value
            for key, value in sample.items()
        }
        return sample
    
    def pad(self, tensor: Tensor) -> Tensor:
        """
        Pad the tensor with 0, to 2209.
        """
        pad_size: int = 2209 - tensor.size(0)
        if pad_size == 0:
            return tensor
        else:
            return torch.nn.functional.pad(tensor, (0, pad_size), "constant", 0)


def main(config: Config):
    # Set device to GPU if available
    if cuda_is_available():
        device = set_device("cuda")
    else:
        device = set_device("cpu")
    print(f"Using device: {device}")

    dataset_config: Config = config.dataset
    dataset: LMDBDataset = LMDBDataset(Path(dataset_config.dataset_name), transform=Transform())
    dataset_length: int = len(dataset)

    test_split: float = dataset_config.test_split
    val_split: float = dataset_config.validation_split

    test_length: int = int(test_split * dataset_length)
    val_length: int = int(val_split * dataset_length)
    train_length: int = dataset_length - test_length - val_length

    train_dataset, test_dataset, val_dataset = random_split(
            dataset,
            [
                train_length,
                val_length,
                test_length
            ]
        )

    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
        collate_fn=dataset.lmdb_collate_fn,
        pin_memory=False,
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        collate_fn=dataset.lmdb_collate_fn,
        pin_memory=True,
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=dataset.lmdb_collate_fn,
        pin_memory=True,
    )

    # Training initalization
    training_config: Config = config.training
    num_epochs: int = training_config.epochs

    criterion_options: dict[str, Type[Module]] = {"mse": MSELoss, "l1": L1Loss}
    criterion: Module = criterion_options[training_config.loss_function]()

    # Logging initialization
    print_points = init_logging(config, train_dataloader)

    # EXPERIMENT LOOP #
    print(
        f"\nStarting experiment: {config.experiment.experiment_name}"
    )

    # Initialize or reset
    model: DynamicModel = init_model(config).to(device)
    optimizer: Optimizer = init_optimizer(config, model)

    figure_folder = config.experiment.output_folder / f"figures"
    model_output_folder = figure_folder / f"model_output"
    model_output_folder.mkdir(parents=True, exist_ok=True)

    logging_params: LoggingParameters = LoggingParameters(
        loss_value=[],
        test_loss_value=[],
        print_points=print_points,
        print_points_list=sorted(list(print_points)),
        batch_size=dataset_config.batch_size,
        figure_folder=figure_folder,
        model_output_folder=model_output_folder,
    )

    # Train
    train(
        num_epochs,
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        criterion,
        device,
        logging_params
    )

    print("Training complete.")


if __name__ == "__main__":
    pass
