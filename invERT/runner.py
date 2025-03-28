import logging
from time import perf_counter
from torch import device as set_device
from torch import save as torch_save
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
from model.train import train, print_model_results
from data.data import (
    generate_data,
    pre_process_data,
    initialize_datasets,
    LMDBDataset,
    lmdb_custom_collate_fn
    pre_process_data_lmdb
)
import matplotlib.pyplot as plt
import multiprocessing as mp
from model.parameters_classes import TestingParameters


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

    in_channels: list[int] = []
    kernel_shapes: list[int] = []
    for conv_layer in cnn_config.conv_layers:
        in_channels.append(conv_layer.in_channels)
        kernel_shapes.append(conv_layer.kernel_shape)

    model = DynamicModel(
        mlp_config.input_metadata_dim,
        mlp_config.hidden_dims,
        in_channels,
        cnn_config.out_channels,
        kernel_shapes
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
            weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.type == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=initial_lr,
            weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.type == "rmsprop":
        optimizer = RMSprop(
            model.parameters(),
            lr=initial_lr,
            weight_decay=optimizer_config.weight_decay)

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
                 train_dataloaders: list[DataLoader] | DataLoader
                 ) -> tuple[int, int, int]:
    logging_config: Config = config.logging
    if isinstance(train_dataloaders, DataLoader):
        nb_batches: int = len(train_dataloaders)
    else:
        nb_batches: int = len(train_dataloaders[0])
    nb_print_points: int = logging_config.print_points
    # Calculate the batch_index at which to print
    print_points: int = nb_batches // logging_config.print_points
    logging.debug(f"nb_batches: {nb_batches}")
    logging.debug(
        f"logging_config.print_points: {logging_config.print_points}")
    logging.debug(f"Print points: {print_points}")

    return print_points, nb_print_points, nb_batches


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
    queue = None
    # queue = mp.Queue()
    # plotter = mp.Process(target=plot_loss, args=(queue,))
    # plotter.start()

    # if not launch_plotter(queue):
    #     print("Failed to start plotter process.")

    # Set device to GPU if available
    if cuda_is_available():
        device = set_device("cuda")
    else:
        device = set_device("cpu")
    print(f"Using device: {device}")

    # Initialize data
    if config.dataset.dataset_name == "_generated":
        data, val_data, target, val_target, input_max_shape, min_data, \
            max_data, min_target, max_target = init_gen_data(config)
        # Initialize dataloaders
        train_dataloaders, test_dataloaders, val_dataloaders = \
            init_gen_dataloaders(config, data, val_data, target, val_target)
    else:
        dataset_config: Config = config.dataset
        dataset: LMDBDataset = LMDBDataset(Path(dataset_config.dataset_name))
        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
            num_workers=8,
            collate_fn=lmdb_custom_collate_fn,
        )
        pre_process_data_lmdb(dataloader)

        train_dataset, test_dataset, val_dataset = dataset.split(
            dataset_config.test_split,
            dataset_config.validation_split
        )
        train_dataloaders: DataLoader = DataLoader(
            train_dataset,
            batch_size=dataset_config.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=lmdb_custom_collate_fn,
        )
        test_dataloaders: DataLoader = DataLoader(
            test_dataset,
            batch_size=dataset_config.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=lmdb_custom_collate_fn,
        )
        val_dataloaders: DataLoader = DataLoader(
            val_dataset,
            batch_size=dataset_config.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=lmdb_custom_collate_fn,
        )

    # Training initalization
    training_config: Config = config.training
    epochs: int = training_config.epochs

    criterion_options: dict = {"mse": MSELoss, "l1": L1Loss}
    criterion = criterion_options[training_config.loss_function]()

    # Logging initialization
    print_points, nb_print_points, nb_batches = \
        init_logging(config, train_dataloaders)

    # Array to store the lresults of each repetition
    loss_arrays: np.ndarray = np.zeros(
        (config.experiment.repetitions, nb_print_points * epochs),
        dtype=float
    )
    test_loss_arrays: np.ndarray = np.zeros(
        (config.experiment.repetitions, nb_print_points * epochs),
        dtype=float)
    model_list: np.ndarray = np.empty(config.experiment.repetitions,
                                      dtype=object)
    optimizer_list: np.ndarray = np.empty(config.experiment.repetitions,
                                          dtype=object)
    scheduler_list: np.ndarray = np.empty(config.experiment.repetitions,
                                          dtype=object)

    # EXPERIMENT LOOP #
    for repetition in range(config.experiment.repetitions):
        print(
            f"\nStarting repetition "
            f"{repetition + 1}/{config.experiment.repetitions} "
            f"of experiment: {config.experiment.experiment_name}")
        # To time the execution of a repetition
        start_time: float = perf_counter()

        # Initialize or reset
        model: DynamicModel = init_model(config).to(device)
        optimizer: Optimizer = init_optimizer(config, model)
        scheduler: Module = init_scheduler(config, optimizer)

        output_folder: Path = Path()
        output_folder = config.experiment.output_folder / \
            f"repetition_{repetition + 1}" / "figures"
        output_folder.mkdir(parents=True, exist_ok=True)

        testing_params: TestingParameters = \
            TestingParameters(loss_arrays,
                              test_loss_arrays,
                              repetition,
                              print_points,
                              nb_print_points,
                              queue,
                              0,
                              0)

        # Train
        model = train(model,
                      epochs,
                      nb_batches,
                      train_dataloaders,
                      test_dataloaders,
                      optimizer,
                      criterion,
                      input_max_shape,
                      device,
                      scheduler,
                      testing_params
                      )

        # print time in hh:mm:ss
        elapsed_time: float = perf_counter() - start_time
        if elapsed_time >= 60:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            formatted_time = f"{minutes:02}min {seconds:02}s"
        else:
            formatted_time = f"{elapsed_time:.2f}s"
        print(f"Repetition {repetition + 1} ended after {formatted_time}.")

        model_list[repetition] = model
        optimizer_list[repetition] = optimizer
        scheduler_list[repetition] = scheduler
    # queue.put("stop")
    # plotter.terminate()

    loss_array_mean = np.mean(loss_arrays, axis=0)
    loss_array_std = np.std(loss_arrays, axis=0)

    test_loss_array_mean = np.mean(test_loss_arrays, axis=0)
    test_loss_array_std = np.std(test_loss_arrays, axis=0)

    np.savez_compressed(
        config.experiment.output_folder / "loss_arrays.npz",
        loss_arrays=loss_arrays,
        loss_array_mean=loss_array_mean,
        loss_array_std=loss_array_std,
        test_loss_arrays=test_loss_arrays,
        test_loss_array_mean=test_loss_array_mean,
        test_loss_array_std=test_loss_array_std)

    # Save the state of the model with the best test loss
    # over the last 10% of the logs
    best_model_index = np.argmin(
        np.mean(test_loss_arrays[-len(test_loss_arrays) // 10:], axis=1)
    )
    state = {
        "model": model_list[best_model_index].state_dict(),
        "optimizer": optimizer_list[best_model_index].state_dict(),
        "scheduler": scheduler_list[best_model_index].state_dict()
    }
    torch_save(state, config.experiment.output_folder / "best_model.pth")

    burn_in: int = 2
    # Plot results
    for repetition in range(config.experiment.repetitions):
        color = plt.cm.tab10(repetition % 10)
        plt.plot(
            loss_arrays[repetition][burn_in:],
            label=f"Repetition {repetition + 1}",
            color=color,
        )
        plt.plot(
            test_loss_arrays[repetition][burn_in:],
            label=f"Repetition {repetition + 1} (test)",
            color=color,
            alpha=0.7
        )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and testing loss")
    plt.legend()
    plt.savefig(config.experiment.output_folder / "training_testing_loss.png")
    # plt.show()

    # Plot and save mean and std train_loss
    plt.figure(figsize=(10, 6))  # Set figure size for better visibility
    plt.plot(
        range(len(loss_array_mean[burn_in:])),
        loss_array_mean[burn_in:],
        label="Training Loss",
        color="blue",
        linewidth=2
    )  # Main line for the mean
    plt.fill_between(
        range(len(loss_array_mean[burn_in:])),
        loss_array_mean[burn_in:] - loss_array_std[burn_in:],
        loss_array_mean[burn_in:] + loss_array_std[burn_in:],
        color="lightblue",
        alpha=0.8,
        label="Standard Deviation"
    )  # Shaded area for standard deviation

    plt.xlabel("Iteration", fontsize=14)  # Increase font size for labels
    plt.ylabel("Loss", fontsize=14)
    plt.title(
        f"Training Loss Mean over {config.experiment.repetitions} Repetitions",
        fontsize=16
    )
    plt.xticks(fontsize=12)  # Adjust font size for tick labels
    plt.yticks(fontsize=12)
    plt.grid(
        True, linestyle="--", alpha=0.8)  # Add a grid for better readability
    plt.legend(fontsize=12)  # Adjust legend font size
    plt.tight_layout()  # Optimize spacing
    plt.savefig(config.experiment.output_folder / "training_loss_mean.png")
    # plt.show()

    # Plot and save mean and std test_loss
    plt.figure(figsize=(10, 6))  # Set figure size for better visibility
    plt.plot(
        range(len(test_loss_array_mean[burn_in:])),
        test_loss_array_mean[burn_in:],
        label="Testing Loss",
        color="blue",
        linewidth=2
    )  # Main line for the mean
    plt.fill_between(
        range(len(test_loss_array_mean[burn_in:])),
        test_loss_array_mean[burn_in:] - test_loss_array_std[burn_in:],
        test_loss_array_mean[burn_in:] + test_loss_array_std[burn_in:],
        color="lightblue",
        alpha=0.8,
        label="Standard Deviation"
    )  # Shaded area for standard deviation

    plt.xlabel("Iteration", fontsize=14)  # Increase font size for labels
    plt.ylabel("Loss", fontsize=14)
    plt.title(
        f"Testing Loss Mean over {config.experiment.repetitions} Repetitions",
        fontsize=16
    )
    plt.xticks(fontsize=12)  # Adjust font size for tick labels
    plt.yticks(fontsize=12)
    plt.grid(
        True, linestyle="--", alpha=0.8)  # Add a grid for better readability
    plt.legend(fontsize=12)  # Adjust legend font size
    plt.tight_layout()  # Optimize spacing
    plt.savefig(config.experiment.output_folder / "testing_loss_mean.png")
    # plt.show()

    print_model_results(model_list,
                        val_dataloaders,
                        input_max_shape,
                        device,
                        min_target,
                        max_target,
                        config.experiment.output_folder)


if __name__ == "__main__":
    pass
