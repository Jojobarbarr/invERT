import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set backend early, prevents GUI issues in non-GUI envs
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from collections import deque

from model.models import DynamicModel
from data.data import InvERTSample
from model.parameters_classes import LoggingParameters

from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

# --- Constants ---
NUM_PLOT_SAMPLES = 3


# --- Helper Functions ---

def tensors_to_numpy_list(tensor_list: list[Tensor]) -> list[np.ndarray]:
    """
    Converts a list of tensors to a list of NumPy arrays,
    filtering out any tensors that are all zeros.

    Args:
        tensor_list: List of tensors (presumably on CPU).

    Returns:
        List of NumPy arrays.
    """
    result = []
    for tensor in tensor_list:
        # Check if tensor contains any non-zero elements before converting
        if torch.any(tensor != 0):
            result.append(tensor.cpu().numpy())
    return result

# --- Plotting Functions ---

def plot_samples(
    psections_np: list[np.ndarray],
    targets_np: list[np.ndarray],
    JtJ_diag_np: list[np.ndarray],
    preds_np: list[np.ndarray],
    num_electrodes_list: list[np.ndarray],
    array_type_list: list[np.ndarray],
    prefix: str,
    current_step: int,
    logging_parameters: LoggingParameters
) -> None:
    # Ensure we have masks corresponding to other data
    num_samples_to_plot = NUM_PLOT_SAMPLES
    num_columns = 5

    fig, axs = plt.subplots(num_samples_to_plot, num_columns, figsize=(4 * num_columns, 4 * num_samples_to_plot), squeeze=False)
    fig.suptitle(f"{prefix} @ Step {current_step}: Pseudosection, Output, Target, Weights, Weighted Error", fontsize=14)

    for i in range(num_samples_to_plot):
        psec_orig = psections_np[i].squeeze() # Use squeeze to remove dims of size 1
        target_orig = targets_np[i].squeeze()
        JtJ_diag_orig = JtJ_diag_np[i].squeeze()
        pred_orig = preds_np[i].squeeze()
        num_electrodes = int(num_electrodes_list[i] * 72 + 24)
        array_type = array_type_list[i]

        # --- 1. Find Bounding Box ---
        valid_values = np.where(JtJ_diag_orig != 0, True, False) # Mask for valid values
        rows, cols = np.where(valid_values)
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()

        # --- 2. Crop Data to Bounding Box ---
        psec_width = num_electrodes - 3
        psec_height = num_electrodes // 2 - 1 if array_type else (num_electrodes - 1) // 3
        psec_cropped = psec_orig[:psec_height, :psec_width]
        psec_cropped = np.where(psec_cropped == 0, np.nan, psec_cropped) # Replace zeros with NaN for visualization

        target_cropped = target_orig[r_min:r_max, c_min:c_max]
        JtJ_diag_cropped = JtJ_diag_orig[r_min:r_max, c_min:c_max]
        pred_cropped = pred_orig[r_min:r_max, c_min:c_max]
        valid_values = valid_values[r_min:r_max, c_min:c_max]

        # --- Calculate error ---
        weighted_error_cropped = JtJ_diag_cropped * (target_cropped - pred_cropped) ** 2

        # --- 4. Plotting Cropped Data ---
        cmap_plots = 'viridis'
        cmap_error = 'magma'

        # Plot Pseudosection
        im0 = axs[i, 0].imshow(psec_cropped, cmap=cmap_plots)
        axs[i, 0].set_title("Pseudosection")
        fig.colorbar(im0, ax=axs[i, 0])

        # vmin = np.nanmin([np.nanmin(target_cropped), np.nanmin(pred_cropped)])
        # vmax = np.nanmax([np.nanmax(target_cropped), np.nanmax(pred_cropped)])
        vmin = 0
        vmax = 1

        # Plot Prediction
        im1 = axs[i, 1].imshow(pred_cropped, cmap=cmap_plots, vmin=vmin, vmax=vmax)
        axs[i, 1].set_title("Output")
        fig.colorbar(im1, ax=axs[i, 1])

        # Plot Target
        im2 = axs[i, 2].imshow(target_cropped, cmap=cmap_plots, vmin=vmin, vmax=vmax)
        axs[i, 2].set_title("Target")
        fig.colorbar(im2, ax=axs[i, 2])

        # Plot JtJ_diag
        JtJ_diag_vmin = np.nanmin(JtJ_diag_cropped)
        JtJ_diag_vmax = np.nanmax(JtJ_diag_cropped)
        im3 = axs[i, 3].imshow(JtJ_diag_cropped, cmap=cmap_plots, vmin=JtJ_diag_vmin, vmax=JtJ_diag_vmax)
        axs[i, 3].set_title("Loss Weights")
        fig.colorbar(im3, ax=axs[i, 3])

        # Plot Error
        error_vmin = np.nanmin(weighted_error_cropped)
        error_vmax = np.nanmax(weighted_error_cropped)
        im4 = axs[i, 4].imshow(weighted_error_cropped, cmap=cmap_error, vmin=error_vmin, vmax=error_vmax)
        axs[i, 4].set_title("Weighted Error")
        fig.colorbar(im4, ax=axs[i, 4])
    fig.tight_layout()
    if prefix == "Train":
        plot_filename = logging_parameters.model_output_folder_train / f"output_step_{current_step}_{prefix.lower()}.png"
    else:
        plot_filename = logging_parameters.model_output_folder_test / f"output_step_{current_step}_{prefix.lower()}.png"

    fig.savefig(plot_filename)
    plt.close(fig)


def plot_loss_curve(logging_parameters: LoggingParameters) -> None:
    """
    Plots the training and testing loss curves.

    Args:
        logging_parameters: Object containing loss history and paths.
    """
    if not logging_parameters.loss_value or not logging_parameters.test_loss_value:
        logging.warning("No loss values recorded. Skipping loss curve plot.")
        return

    # Use the recorded batch indices for the x-axis
    steps = logging_parameters.print_points_list[:len(logging_parameters.loss_value)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, logging_parameters.loss_value, label="Train Loss", marker='o', color='b')
    ax.plot(steps, logging_parameters.running_loss_value[:len(steps)], label="Running Loss", marker='s', color='g')
    ax.plot(steps, logging_parameters.test_loss_value[:len(steps)], label="Test Loss", marker='x', color='r')
    ax.set_xlabel("Training Step (Batch Index)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Training Step")
    ax.legend()
    ax.grid(True)

    loss_filename = logging_parameters.figure_folder / "loss_curve.png"
    fig.savefig(loss_filename)
    plt.close(fig)


# --- Evaluation Function ---

def evaluate(
    model: DynamicModel,
    dataloader: DataLoader,
    criterion: Module,
    device: torch.device,
    use_amp: bool
) -> float:
    """
    Evaluates the model on the provided dataloader.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the evaluation dataset.
        criterion: The loss function (expecting reduction='none').
        device: The device to run evaluation on.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Average loss over the entire dataset.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_elements = 0

    with torch.no_grad(): # Disable gradient calculations
        for batch in tqdm(dataloader, desc="Evaluating", leave=False, unit="batch"):
            pseudosection_batch = batch['pseudosection'].to(device)
            target_batch = batch['norm_log_resistivity_model'].to(device)
            weights_matrix = batch['JtJ_diag'].to(device)

            with autocast(device.type, enabled=use_amp):
                output_batch = model(pseudosection_batch, target_batch)
                weighted_loss = weights_matrix * criterion(output_batch, target_batch)

            # batch loss is sum of all loss per pixel divided by the weights on those pixels.
            batch_loss_sum = weighted_loss.sum()
            # weights_sum = weights_matrix.sum()
            weights_sum = torch.sum(weights_matrix != 0).item() # Count non-zero weights

            if weights_sum > 0:
                total_loss += batch_loss_sum.item()
                total_elements += weights_sum.item()

    model.train() # Set model back to training mode
    if total_elements == 0:
        logging.warning("No valid elements found during evaluation. Returning 0 loss.")
        return 0.0
    return total_loss / total_elements


# --- Training Step Functions ---

def process_batch(
    batch: dict[str, Tensor], # Assuming InvERTSample is dict-like
    model: DynamicModel,
    criterion: Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool
) -> float:
    """
    Processes a single batch for training (forward, backward, optimization).

    Args:
        batch: A dictionary containing the batch data.
        model: The model to train.
        criterion: The loss function (expecting reduction='none').
        optimizer: The optimizer.
        scaler: GradScaler for mixed precision.
        device: The device being used.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        The calculated loss for this batch.
    """
    pseudosection_batch = batch['pseudosection']
    target_batch = batch['norm_log_resistivity_model']
    weights_matrix = batch['JtJ_diag']

    optimizer.zero_grad()

    with autocast(device.type, enabled=use_amp):
        output_batch = model(pseudosection_batch, target_batch)
        weighted_loss = weights_matrix * criterion(output_batch, target_batch) # reduction='none'

        # Apply mask and calculate effective batch loss
        # weights_sum = weights_matrix.sum()
        weights_sum = torch.sum(weights_matrix != 0).item() # Count non-zero weights

        if weights_sum > 0:
            batch_loss = weighted_loss.sum() / weights_sum
        else:
            # Handle cases with no valid targets in a batch if necessary
            raise ValueError("No valid elements in batch for loss calculation.")

    # Scale the loss, perform backward pass, and update optimizer
    if use_amp:
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        batch_loss.backward()
        optimizer.step()
    
    if scheduler is not None:
        scheduler.step()

    return batch_loss.item()


def process_epoch(
    model: DynamicModel,
    running_loss: float,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler,
    criterion: Module,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    logging_parameters: LoggingParameters
) -> None:
    model.train() # Ensure model is in training mode

    try:
        persistent_test_batch = next(iter(test_dataloader))
        persistent_test_batch = {k: v.to(device) for k, v in persistent_test_batch.items()}
    except StopIteration:
        logging.warning("Test dataloader is empty. Cannot visualize test samples.")
        persistent_test_batch = None

    for batch_idx, batch in tqdm(
        enumerate(train_dataloader),
        desc=f"Epoch {epoch} Training",
        total=len(train_dataloader),
        unit="batch"
    ):
        # Move training batch data to the target device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Process the batch
        batch_loss = process_batch(
            batch, model, criterion, optimizer, scheduler, scaler, device, use_amp
        )

        # Update running loss
        running_loss.append(batch_loss)

        # --- Logging and Evaluation ---
        if batch_idx in logging_parameters.print_points or batch_idx == len(train_dataloader) - 1:
            current_step = (epoch - 1) * len(train_dataloader) + batch_idx # Global step

            # 1. Calculate Average Test Loss over the entire test set
            avg_test_loss = evaluate(model, test_dataloader, criterion, device, use_amp)

            running_loss_average = sum(running_loss) / len(running_loss)

            # 2. Log losses
            logging_parameters.loss_value.append(batch_loss) # Log instantaneous train loss
            logging_parameters.running_loss_value.append(running_loss_average) # Log running average train loss
            logging_parameters.test_loss_value.append(avg_test_loss)

            # Keep track of which batch index this corresponds to
            logging_parameters.print_points_list.append(current_step) # Log global step

            print(
                f"\nEpoch {epoch} | Batch {batch_idx}/{len(train_dataloader)} | Step {current_step} | "
                f"Train Loss (batch): {batch_loss:.5f} | "
                f"Running Loss (avg): {running_loss_average:.5f} | "
                f"Test Loss (avg): {avg_test_loss:.5f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            # 3. Plot sample predictions (using the persistent test batch)
            if persistent_test_batch:
                model.eval() # Switch to eval mode for prediction
                with torch.no_grad():
                    with autocast(device.type, enabled=use_amp):
                        test_outputs = model(persistent_test_batch['pseudosection'], persistent_test_batch['norm_log_resistivity_model'])
                model.train() # Switch back

                # Prepare data for plotting (move needed tensors to CPU)
                psec_test_np = tensors_to_numpy_list(persistent_test_batch['pseudosection'].cpu())
                target_test_np = tensors_to_numpy_list(persistent_test_batch['norm_log_resistivity_model'].cpu())
                JtJ_diag_test_np = tensors_to_numpy_list(persistent_test_batch['JtJ_diag'].cpu())
                output_test_np = tensors_to_numpy_list(test_outputs.cpu())
                test_num_electrodes_list = persistent_test_batch['num_electrode'].cpu().numpy()
                test_array_type_list = [torch.argmax(array_type).cpu().numpy() for array_type in persistent_test_batch['array_type']]

                # Also plot the current training batch examples
                psec_train_np = tensors_to_numpy_list(batch['pseudosection'].cpu())
                target_train_np = tensors_to_numpy_list(batch['norm_log_resistivity_model'].cpu())
                JtJ_diag_train_np = tensors_to_numpy_list(batch['JtJ_diag'].cpu())
                train_num_electrodes_list = batch['num_electrode'].cpu().numpy()
                train_array_type_list = [torch.argmax(array_type).cpu().numpy() for array_type in batch['array_type']]

                # Need to recompute output for the training batch
                model.eval()
                with torch.no_grad():
                    with autocast(device.type, enabled=use_amp):
                        train_outputs_plot = model(batch['pseudosection'], batch['norm_log_resistivity_model'])
                model.train()
                output_train_np = tensors_to_numpy_list(train_outputs_plot.cpu())


                plot_samples(
                    psec_test_np, target_test_np, JtJ_diag_test_np, output_test_np, test_num_electrodes_list, test_array_type_list,
                    "Test", current_step, logging_parameters
                )
                plot_samples(
                    psec_train_np, target_train_np, JtJ_diag_train_np, output_train_np, train_num_electrodes_list, train_array_type_list,
                    "Train", current_step, logging_parameters
                )

            # 4. Plot overall loss curve
            plot_loss_curve(logging_parameters)


# --- Main Training Function ---

def train(
    num_epochs: int,
    model: DynamicModel,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: Module,
    device: torch.device,
    use_amp: bool,
    logging_parameters: LoggingParameters
):
    """
    Main training loop for the model.

    Args:
        num_epochs: Number of epochs to train for.
        model: The PyTorch model to train.
        train_dataloader: DataLoader for the training data.
        test_dataloader: DataLoader for the test data.
        optimizer: The optimizer for updating model weights.
        criterion: The loss function. MUST have reduction='none' for masking.
        device_name: Name of the device to use ('cuda', 'cpu', etc.).
        use_amp: Whether to use Automatic Mixed Precision (requires CUDA).
        logging_parameters: Configuration object for logging results.
    """
    assert criterion.reduction == 'none', "Criterion must have reduction='none' for masking."

    model.to(device)
    print(f"\nUsing device: {device}")

    # AMP setup
    amp_enabled = use_amp and (device.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled)
    print(f"Automatic Mixed Precision (AMP): {'Enabled' if amp_enabled else 'Disabled'}")

    # --- Training Loop ---
    print(f"Starting training for {num_epochs} epochs...")
    running_loss = deque(maxlen=100)
    
    for epoch in range(1, num_epochs + 1): # Start epochs from 1
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")

        process_epoch(
            model,
            running_loss,
            train_dataloader,
            test_dataloader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            device,
            amp_enabled,
            epoch,
            logging_parameters
        )

        checkpoint_path = logging_parameters.checkpoint_folder / f"model_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': logging_parameters.test_loss_value[-1] if logging_parameters.test_loss_value else None,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


    print("\n--- Training Finished ---")
    if logging_parameters.loss_value and logging_parameters.test_loss_value:
        # current_step = epoch * len(train_dataloader)
        print(f"Final Train Loss (last logged): {logging_parameters.loss_value[-1]:.5f}")
        print(f"Final Running Loss (last avg): {logging_parameters.running_loss_value[-1]:.5f}")
        print(f"Final Test Loss (last avg): {logging_parameters.test_loss_value[-1]:.5f}")
    else:
        print("No loss values were logged during training.")
