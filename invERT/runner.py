from pathlib import Path
import multiprocessing as mp
from typing import Type, Optional

import torch
from torch import device as set_device
from torch.cuda import is_available as cuda_is_available
from torch.optim import AdamW, SGD, RMSprop
from torch.optim.optimizer import Optimizer
from torch.nn import Module, MSELoss, L1Loss
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
    OneCycleLR,
    _LRScheduler # Base class for type hinting
)

from config.configuration import Config
from model.models import UNet, UNetAttention, UNet_basic
from model.train import train
from model.parameters_classes import LoggingParameters
from data.data import (
    InvERTDataset,
    InvERTSample,
    collate_pad,
    collate_per_sample
)


def init_model(config) -> Module:
    model = UNet()
    # model = VAE()
    # model = UNetAttention()
    # model = MLP_huge()
    return model


def init_optimizer(config: Config,
                   model: Module
                   ) -> Optimizer:
    initial_lr: float = config.training.learning_rate

    optimizer_config: Config = config.training.optimizer

    optimizer: Optimizer
    if optimizer_config.type == "adamw":
        optimizer = AdamW(
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
                   optimizer: Optimizer,
                   train_dataloader_len: int # Pass length for step calculations
                   ) -> Optional[_LRScheduler]: # Use base class for return type hint

    scheduler_config: Config = config.training.scheduler
    num_epochs = config.training.epochs

    if not scheduler_config.enabled:
        print("LR Scheduler: Disabled")
        return None

    lr_scheduler_type = scheduler_config.type
    print(f"LR Scheduler: Enabling {lr_scheduler_type}")

    scheduler: Optional[_LRScheduler] = None

    if lr_scheduler_type == "plateau":
        # Your existing ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=1,
            verbose=True,
        )
        print(f"  - Mode: {scheduler.mode}, Factor: {scheduler.factor}, Patience: {scheduler.patience}")
        # Note: ReduceLROnPlateau typically steps based on validation loss after an epoch.

    elif lr_scheduler_type == "cosine_warmup":
        # Cosine Annealing with Linear Warmup using SequentialLR
        eta_min = 1e-5 # Minimum LR for cosine decay
        warmup_steps = int(train_dataloader_len * 0.3)
        cosine_steps = int(train_dataloader_len * 2)

        print(f"  - Warmup Steps: ({warmup_steps} steps)")
        print(f"  - Cosine Decay Steps: {cosine_steps}")
        print(f"  - Eta Min: {eta_min}")

        # Scheduler 1: Linear Warmup
        # Starts from a factor (e.g., 0.01) and goes up to 1.0 over warmup_steps
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Scheduler 2: Cosine Decay
        # T_max is the number of steps for the decay phase
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_steps,
            T_mult=2,
            eta_min=eta_min
        )

        # Combine them: Use warmup for `warmup_steps`, then cosine for the rest
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps] # The step count at which to switch
        )
        # Note: SequentialLR requires stepping after *each batch/iteration*.

    elif lr_scheduler_type == "onecycle":
        # OneCycleLR scheduler
        max_lr = optimizer.param_groups[0]['lr'] # Use optimizer's initial LR as max_lr
        # Or get max_lr explicitly from config: scheduler_config.max_lr
        total_steps = num_epochs * train_dataloader_len
        pct_start = 0.3 # Percentage of cycle for warmup
        anneal_strategy = 'cos' # 'cos' or 'linear'
        final_div_factor = 1e4 # LR decays to max_lr / div_factor / final_div_factor

        print(f"  - Max LR: {max_lr}")
        print(f"  - Total Steps: {total_steps}")
        print(f"  - Warmup Fraction (pct_start): {pct_start}")
        print(f"  - Anneal Strategy: {anneal_strategy}")


        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            final_div_factor=final_div_factor,
            # You might want to configure momentum cycling if using SGD w/ momentum
            # cycle_momentum=scheduler_config.get("cycle_momentum", True), # Example
        )
        # Note: OneCycleLR requires stepping after *each batch/iteration*.

    else:
        print(f"LR Scheduler: Type '{lr_scheduler_type}' not recognized or explicitly disabled.")
        return None # Return None if type not found or disabled

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
        for i in range(0, num_print_points)
    }

    return print_points



class Transform:
    def __init__(self):
        pass

    def __call__(self, sample: InvERTSample) -> InvERTSample:
        pseudosection = sample['pseudosection'].unsqueeze(0)
        num_electrode_channel = torch.ones_like(pseudosection) * sample['num_electrode']
        subsection_length_channel = torch.ones_like(pseudosection) * sample['subsection_length']
        array_type_channel = sample['array_type']
        array_type_channel = array_type_channel.view(-1, 1, 1).expand(-1, pseudosection.shape[1], pseudosection.shape[2])
        sample['pseudosection'] = torch.cat((pseudosection, num_electrode_channel, subsection_length_channel, array_type_channel), dim=0)
        
        sample['norm_log_resistivity_model'] = sample['norm_log_resistivity_model'].unsqueeze(0)
        sample['JtJ_diag'] = sample['JtJ_diag'].unsqueeze(0)
        # sample['JtJ_diag'] = torch.ones_like(sample['JtJ_diag'])
        
        return sample


def main(config: Config):
    mp.set_start_method("spawn")
    # Set device to GPU if available
    if cuda_is_available():
        device = set_device("cuda")
    else:
        device = set_device("cpu")
    print(f"Using device: {device}")

    
    print(f"Initializing model")
    model: Module = init_model(config).to(device)

    dataset_config: Config = config.dataset
    dataset: InvERTDataset = InvERTDataset(Path(dataset_config.dataset_name), transform=Transform())
    # dataset = dataset[:128]
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
        prefetch_factor=2,
        collate_fn=collate_pad,
        pin_memory=True,
        persistent_workers=True,
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=False,
        num_workers=8,
        prefetch_factor=2,
        collate_fn=collate_pad,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=False,
        num_workers=8,
        prefetch_factor=1,
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
    # criterion = WeightedMSEWithVariationLoss(0.25)

    # Logging initialization
    print_points = init_logging(config, train_dataloader)

    # Initialize or reset
    print(f"Initializing optimizer")
    optimizer: Optimizer = init_optimizer(config, model)
    scheduler: Optional[_LRScheduler] = init_scheduler(config, optimizer, len(train_dataloader))

    figure_folder = config.experiment.output_folder / f"figures"
    model_output_folder = figure_folder / f"model_output"
    model_output_folder_train = model_output_folder / f"train"
    model_output_folder_test = model_output_folder / f"test"
    checkpoint_folder = model_output_folder / f"checkpoints"
    validation_folder = figure_folder / f"validation"
    model_output_folder_train.mkdir(parents=True, exist_ok=True)
    model_output_folder_test.mkdir(parents=True, exist_ok=True)
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    figure_folder.mkdir(parents=True, exist_ok=True)
    validation_folder.mkdir(parents=True, exist_ok=True)

    logging_params: LoggingParameters = LoggingParameters(
        loss_value=[],
        running_loss_value=[],
        test_loss_value=[],
        print_points=print_points,
        print_points_list=[],
        batch_size=dataset_config.batch_size,
        figure_folder=figure_folder,
        model_output_folder_train=model_output_folder_train,
        model_output_folder_test=model_output_folder_test,
        checkpoint_folder=checkpoint_folder,
        validation_folder=validation_folder,
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
        val_dataloader,
        optimizer,
        scheduler,
        criterion,
        device,
        use_amp=True,
        logging_parameters=logging_params,
    )

    print("Training complete.")


if __name__ == "__main__":
    pass
