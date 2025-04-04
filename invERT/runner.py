from torch import device as set_device
from torch import Tensor
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available
from torch.optim import Adam, SGD, RMSprop
from torch.optim.optimizer import Optimizer
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss, L1Loss
import numpy as np
from config.configuration import Config
from model.models import DynamicModel
from pathlib import Path
from model.train import train
from torch.utils.data import random_split
from data.data import (
    generate_data,
    pre_process_data,
    initialize_datasets,
    LMDBDataset,
    lmdb_custom_collate_fn
)
from typing import Type
import matplotlib.pyplot as plt
import multiprocessing as mp
from model.parameters_classes import LoggingParameters


def init_gen_data(dataset_config: Config):
    # The dataset is generated
    num_samples: int = dataset_config.num_samples
    num_sub_groups: int = dataset_config.num_sub_group
    data_min_size: int = dataset_config.data_min_size
    data_max_size: int = dataset_config.data_max_size
    noise: float = dataset_config.noise

    num_samples = int(num_samples * (1 + dataset_config.test_split))
    num_val_samples = int(num_samples * dataset_config.validation_split)
    data, val_data = generate_data(
        num_samples,
        num_sub_groups,
        data_min_size,
        data_max_size,
        noise,
        num_val_samples
    )
    max_input_shape: int = dataset_config.data_max_size
    data, target, min_data, max_data, min_target, max_target = \
        pre_process_data(data)
    val_data, val_target, _, _, _, _ = pre_process_data(val_data)

    return (data,
            val_data,
            target,
            val_target,
            max_input_shape,
            min_data,
            max_data,
            min_target,
            max_target)


def init_gen_dataloaders(config: Config,
                         data: list[Tensor],
                         val_data: list[Tensor],
                         target: list[Tensor],
                         val_target: list[Tensor],
                         ) -> tuple[list[DataLoader]]:
    dataset_config: Config = config.dataset
    batch_size: int = dataset_config.batch_size
    batch_mixture: int = dataset_config.batch_mixture
    num_sub_group: int = dataset_config.num_sub_group
    sub_group_size: int = dataset_config.num_samples // num_sub_group
    test_split: float = dataset_config.test_split

    train_dataloaders, test_dataloaders, val_dataloaders = \
        initialize_datasets(
            data,
            val_data,
            target,
            val_target,
            batch_size,
            batch_mixture,
            num_sub_group,
            sub_group_size,
            test_split,
        )

    return train_dataloaders, test_dataloaders, val_dataloaders


def init_model(config: Config) -> DynamicModel:
    mlp_config: Config = config.model.mlp
    cnn_config: Config = config.model.cnn

    layer_types: list[str] = []
    num_in_channels: list[int] = []
    kernel_shapes: list[int] = []
    strides: list[int] = []
    paddings: list[int] = []

    for layer in cnn_config.conv_layers:
        layer_types.append(layer.layer_type)
        num_in_channels.append(layer.num_in_channels)
        kernel_shapes.append((layer.kernel_width, layer.kernel_height))
        strides.append((layer.stride_width, layer.stride_height))
        if layer.layer_type != "maxpool":
            paddings.append((layer.padding_width, layer.padding_height))
        else:
            paddings.append((0, 0))
    
    num_out_channels: int = cnn_config.num_out_channels
        

    model = DynamicModel(
        mlp_config.input_metadata_dim,
        mlp_config.hidden_dims,
        layer_types,
        num_in_channels,
        kernel_shapes,
        strides,
        paddings,
        num_out_channels,
    )

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
        i * step_between_print
        for i in range(1, num_print_points)
    }
    if num_batches - 1 not in print_points:
        print_points.add(len(train_dataloader) - 1)

    return print_points


def plot_loss(queue: mp.Queue):
    plt.ion()
    fig, ax = plt.subplots()
    print_points = []
    losses = []
    test_losses = []
    queue.put("OK")
    while True:
        if not queue.empty():
            if queue.get() == "stop":
                break
        while not queue.empty():
            loss, test_loss, repetition, print_point = queue.get()
            losses.append(loss)
            test_losses.append(test_loss)
            print_points.append(print_point)
            ax.clear()
            ax.plot(print_points, losses, label="Train loss")
            ax.plot(print_points, test_losses, label="Test loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            plt.pause(0.01)


def launch_plotter(queue: mp.Queue) -> bool:
    while True:
        if not queue.empty():
            if queue.get() == "OK":
                print("Plot process started and ready.")
                return True


def main(config: Config):
    # Set device to GPU if available
    if cuda_is_available():
        device = set_device("cuda")
    else:
        device = set_device("cpu")
    print(f"Using device: {device}")

    dataset_config: Config = config.dataset
    dataset: LMDBDataset = LMDBDataset(Path(dataset_config.dataset_name))
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
        collate_fn=lmdb_custom_collate_fn,
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=lmdb_custom_collate_fn,
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=lmdb_custom_collate_fn,
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

    testing_params: LoggingParameters = LoggingParameters(
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
        testing_params
    )

    print("Training complete.")


if __name__ == "__main__":
    pass
