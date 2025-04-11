from pathlib import Path
import multiprocessing as mp
from typing import Type

from torch import device as set_device
from torch.cuda import is_available as cuda_is_available
from torch.optim import Adam, SGD, RMSprop
from torch.optim.optimizer import Optimizer
from torch.nn import Module, MSELoss, L1Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from config.configuration import Config
from model.models import UNet, MLP_huge
from model.train import train
from model.parameters_classes import LoggingParameters
from data.data import (
    LMDBDataset,
    InvERTSample,
    collate_pad,
    lmdb_collate_fn,
    lmdb_collate_fn_per_cat,
    worker_init_fn,
)


def init_model() -> Module:
    model = UNet()
    # model = MLP_huge()
    return model


def init_optimizer(config: Config,
                   model: Module
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
        sample['pseudosection'] = sample['pseudosection'].unsqueeze(0)
        sample['norm_log_resistivity_model'] = sample['norm_log_resistivity_model'].unsqueeze(0)
        return sample


def main(config: Config):
    mp.set_start_method("spawn")
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
                test_length,
                val_length
            ]
        )

    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=worker_init_fn,
        prefetch_factor=8,
        collate_fn=collate_pad,
        pin_memory=True,
        persistent_workers=True,
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=worker_init_fn,
        prefetch_factor=8,
        collate_fn=collate_pad,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=worker_init_fn,
        prefetch_factor=8,
        collate_fn=collate_pad,
        pin_memory=True,
        persistent_workers=True,
    )
    print(
        f"Dataset initialized with {dataset_length} samples,\n"
        f"Dataloader initialized with {len(train_dataloader)} batches."
    )

    # Training initalization
    training_config: Config = config.training
    num_epochs: int = training_config.epochs

    criterion_options: dict[str, Type[Module]] = {"mse": MSELoss, "l1": L1Loss}
    criterion: Module = criterion_options[training_config.loss_function](reduction="none")

    # Logging initialization
    print_points = init_logging(config, train_dataloader)

    # Initialize or reset
    print(f"Initializing model and optimizer")
    model: Module = init_model(config).to(device)
    optimizer: Optimizer = init_optimizer(config, model)

    figure_folder = config.experiment.output_folder / f"figures"
    model_output_folder = figure_folder / f"model_output"
    model_output_folder_train = model_output_folder / f"train"
    model_output_folder_test = model_output_folder / f"test"
    checkpoint_folder = model_output_folder / f"checkpoints"
    model_output_folder_train.mkdir(parents=True, exist_ok=True)
    model_output_folder_test.mkdir(parents=True, exist_ok=True)
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    figure_folder.mkdir(parents=True, exist_ok=True)

    logging_params: LoggingParameters = LoggingParameters(
        loss_value=[],
        test_loss_value=[],
        print_points=print_points,
        print_points_list=sorted(list(print_points)),
        batch_size=dataset_config.batch_size,
        figure_folder=figure_folder,
        model_output_folder_train=model_output_folder_train,
        model_output_folder_test=model_output_folder_test,
    )

    # model = torch.compile(model) # /!\ Do not work with dynamic model or GPU too old !!!
    # Train
    print(
        f"\nStarting training"
    )
    train(
        num_epochs,
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        criterion,
        device.type,
        use_amp=True,
        logging_parameters=logging_params,
    )

    print("Training complete.")


if __name__ == "__main__":
    pass
