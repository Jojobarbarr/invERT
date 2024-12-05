import logging
from time import perf_counter
from torch import device as set_device
from torch.cuda import is_available as cuda_is_available
from torch.optim import Adam, SGD, RMSprop
from torch.optim.optimizer import Optimizer
from torch.nn import Module
from torch.nn import MSELoss, L1Loss
import numpy as np
from config.configuration import Config
from model.models import DynamicModel
from model.train import train, print_model_results
from data.data import generate_data, pre_process_data, initialize_datasets
import matplotlib.pyplot as plt


def init(config: Config,
         metadata_size: int,
         hidden_layers: list[int],
         num_filters: list[int],
         kernel_sizes: list[int],
         in_channels: int,
         device: str) -> tuple[DynamicModel,
                               Optimizer,
                               Module]:
    model = DynamicModel(
        metadata_size,
        hidden_layers,
        num_filters,
        kernel_sizes,
        in_channels)
    model = model.to(device)

    # Training
    training_config: Config = config.training

    initial_lr: float = training_config.initial_learning_rate

    optimizer_options: dict = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
    optimizer = optimizer_options[training_config.optimizer](
        model.parameters(), lr=initial_lr)

    if training_config.lr_scheduler.enabled:
        lr_scheduler_config: Config = training_config.lr_scheduler
        lr_scheduler_type = lr_scheduler_config.type
        if lr_scheduler_type == "plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=lr_scheduler_config.factor,
                patience=lr_scheduler_config.patience)

    return model, optimizer, scheduler


def main(config: Config):
    if cuda_is_available():
        device = set_device("cuda")
    else:
        device = set_device("cpu")

    # Data
    dataset_config: Config = config.dataset
    if dataset_config.dataset_name == "_generated":
        min_shape: int = dataset_config.data_min_size
        max_shape: int = dataset_config.data_max_size
        num_samples: int = dataset_config.num_samples
        noise: float = dataset_config.noise

        data = generate_data(num_samples, min_shape, max_shape, noise)

    data, target, max_input_shape, min_data, max_data, min_target, max_target = pre_process_data(
        data)

    train_dataloader, test_dataloader, val_dataloader = initialize_datasets(
        data, target, dataset_config.batch_size, dataset_config.test_split, dataset_config.validation_split)

    # Model
    mlp_config: Config = config.model.mlp
    cnn_config: Config = config.model.cnn

    num_filters: list[int] = []
    kernel_sizes: list[int] = []
    for conv_layer in cnn_config.conv_layers:
        # /!\ last filter number must be 1
        num_filters.append(conv_layer.filters)
        kernel_sizes.append(conv_layer.kernel_size)

    # Training
    training_config: Config = config.training
    epochs: int = training_config.epochs

    criterion_options: dict = {"mse": MSELoss, "l1": L1Loss}
    criterion = criterion_options[training_config.loss_function]()

    model, optimizer, scheduler = init(config, mlp_config.input_size, mlp_config.hidden_layers,
                                       num_filters, kernel_sizes, cnn_config.input_channels, device)

    # Print and logging
    logging_config: Config = config.logging
    print_points: int = len(train_dataloader) // logging_config.print_points
    total_print_points: int = logging_config.print_points + \
        ((len(train_dataloader) % logging_config.print_points) // print_points)
    logging.debug(f"len(train_dataloader): {len(train_dataloader)}")
    logging.debug(
        f"logging_config.print_points: {logging_config.print_points}")
    logging.debug(f"Print points: {print_points}")

    loss_array: np.ndarray[np.ndarray[float]] = np.zeros(
        (config.experiment.repetitions, total_print_points))
    test_loss_array: np.ndarray[np.ndarray[float]] = np.zeros(
        (config.experiment.repetitions, total_print_points))
    model_list: list[DynamicModel] = []

    ##### EXPERIMENT LOOP #####
    for repetition in range(config.experiment.repetitions):
        print(
            f"\nStarting repetition {repetition + 1}/{config.experiment.repetitions} of experiment: {config.experiment.experiment_name}")
        model, optimizer, scheduler = init(
            config, mlp_config.input_size, mlp_config.hidden_layers, num_filters, kernel_sizes, cnn_config.input_channels, device)
        loss_array[repetition], test_loss_array[repetition], model = train(
            model, epochs, train_dataloader, test_dataloader, optimizer, criterion, scheduler, loss_array[repetition], test_loss_array[repetition], max_input_shape, device, print_points)
        model_list.append(model)

    for repetition in range(config.experiment.repetitions):
        plt.plot(loss_array[repetition], label=f"Repetition {repetition + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.show()

    print_model_results(
        model_list,
        val_dataloader,
        device,
        max_input_shape,
        min_data,
        max_data,
        min_target,
        max_target)


if __name__ == "__main__":
    pass
