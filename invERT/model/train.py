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

from model.models import DynamicModel
from data.data import InvERTSample
from model.parameters_classes import LoggingParameters

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
    preds_np: list[np.ndarray],
    masks_np: list[np.ndarray],
    num_electrodes_list: list[np.ndarray],
    array_type_list: list[np.ndarray],
    prefix: str,
    current_step: int,
    logging_parameters: LoggingParameters
) -> None:
    """
    Plots pseudosections, targets, predictions, and errors for a few samples,
    masking out padded areas in target, prediction, and error plots.

    Args:
        psections_np: List of NumPy arrays for pseudosections.
        targets_np: List of NumPy arrays for ground truth targets.
        preds_np: List of NumPy arrays for model predictions.
        masks_np: List of NumPy arrays (boolean/int) for the target masks.
        prefix: String prefix for plot title and filename (e.g., "Train", "Test").
        current_step: Current epoch or batch index for unique filenames.
        logging_parameters: Object containing logging paths.
    """
    # Ensure we have masks corresponding to other data
    num_samples_to_plot = min(len(psections_np), len(targets_np), len(preds_np), len(masks_np), NUM_PLOT_SAMPLES)
    if num_samples_to_plot == 0:
        logging.warning(f"No valid samples (with masks) to plot for prefix '{prefix}'. Skipping plot.")
        return

    fig, axs = plt.subplots(num_samples_to_plot, 4, figsize=(16, 4 * num_samples_to_plot), squeeze=False)
    fig.suptitle(f"{prefix} @ Step {current_step}: Pseudosection, Output (Masked), Target (Masked), Error (Masked)", fontsize=14)

    # print(f"Plotting {prefix} - Sample 0 Shapes: Psec={psections_np[0].shape}, Target={targets_np[0].shape}, Pred={preds_np[0].shape}, Mask={masks_np[0].shape}")

    for i in range(num_samples_to_plot):
        # Assuming original tensors might have had extra batch/channel dims removed by [0, 0]
        # Adjust indexing if tensors_to_numpy_list behaves differently or shapes vary
        try:
            psec_orig = psections_np[i].squeeze() # Use squeeze to remove dims of size 1
            target_orig = targets_np[i].squeeze()
            pred_orig = preds_np[i].squeeze()
            mask_orig = masks_np[i].squeeze().astype(bool) # Ensure mask is boolean and squeezed
            num_electrodes = num_electrodes_list[i]
            array_type = array_type_list[i]
        except IndexError:
             logging.warning(f"Skipping plot for sample {i} in {prefix} due to unexpected shape or missing data.")
             continue

        # --- 1. Find Bounding Box ---
        rows, cols = np.where(mask_orig)
        if rows.size == 0: # Mask is all False
            logging.warning(f"Skipping plot for sample {i} in {prefix} because mask is empty.")
            continue
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()

        # --- 2. Crop Data to Bounding Box ---
        psec_width = num_electrodes - 3
        psec_height = num_electrodes // 2 - 1 if array_type else (num_electrodes - 1) // 3
        psec_cropped = psec_orig[:psec_height, :psec_width]
        psec_cropped = np.where(psec_cropped == 0, np.nan, psec_cropped) # Replace zeros with NaN for visualization
        target_cropped = target_orig[r_min:r_max, c_min:c_max]
        pred_cropped = pred_orig[r_min:r_max, c_min:c_max]
        mask_cropped = mask_orig[r_min:r_max, c_min:c_max] # Crucial: use cropped mask too

        # Check if cropping resulted in empty arrays (e.g., if mask was just 1 pixel)
        if psec_cropped.size == 0 or target_cropped.size == 0 or pred_cropped.size == 0 or mask_cropped.size == 0:
             logging.warning(f"Skipping plot for sample {i} in {prefix} due to empty array after cropping.")
             continue


        # --- Calculate vmin/vmax based on ORIGINAL valid data ---
        valid_target_pixels = target_orig[mask_orig]
        valid_pred_pixels = pred_orig[mask_orig]
        if valid_target_pixels.size == 0 or valid_pred_pixels.size == 0: # Should have been caught by rows.size==0, but double check
            continue

        try:
            vmin = np.nanmin([np.nanmin(valid_target_pixels), np.nanmin(valid_pred_pixels)])
            vmax = np.nanmax([np.nanmax(valid_target_pixels), np.nanmax(valid_pred_pixels)])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                 vmin = np.nanmin(valid_target_pixels)
                 vmax = np.nanmax(valid_target_pixels) + 1e-6
                 if not np.isfinite(vmin) or not np.isfinite(vmax):
                    logging.warning(f"Skipping plot for sample {i} in {prefix} due to invalid vmin/vmax on original data.")
                    continue
        except ValueError:
             logging.warning(f"Skipping plot for sample {i} in {prefix} due to empty original data after masking for vmin/vmax.")
             continue

        # --- Calculate error using CROPPED data ---
        error_cropped = (target_cropped - pred_cropped) ** 2

        # --- 3. Apply Mask within Bounding Box (using NaN) ---
        target_cropped_nan = target_cropped.copy().astype(float)
        pred_cropped_nan = pred_cropped.copy().astype(float)
        error_cropped_nan = error_cropped.copy().astype(float)

        target_cropped_nan[~mask_cropped] = np.nan # Use the CROPPED mask
        pred_cropped_nan[~mask_cropped] = np.nan
        error_cropped_nan[~mask_cropped] = np.nan

        # --- 4. Plotting Cropped Data ---
        cmap_plots = 'viridis'
        cmap_error = 'magma'

        # Plot Pseudosection (cropped only)
        im0 = axs[i, 0].imshow(psec_cropped, cmap=cmap_plots)
        axs[i, 0].set_title("Psec (Crop)")
        fig.colorbar(im0, ax=axs[i, 0])

        # Plot Prediction (cropped and NaN-masked)
        im1 = axs[i, 1].imshow(pred_cropped_nan, cmap=cmap_plots, vmin=vmin, vmax=vmax)
        axs[i, 1].set_title("Output (Crop)")
        fig.colorbar(im1, ax=axs[i, 1])

        # Plot Target (cropped and NaN-masked)
        im2 = axs[i, 2].imshow(target_cropped_nan, cmap=cmap_plots, vmin=vmin, vmax=vmax)
        axs[i, 2].set_title("Target (Crop)")
        fig.colorbar(im2, ax=axs[i, 2])

        # Plot Error (cropped and NaN-masked)
        valid_error_pixels_cropped = error_cropped_nan[mask_cropped] # Use cropped mask
        error_vmin = np.nanmin(valid_error_pixels_cropped) if valid_error_pixels_cropped.size > 0 else 0
        error_vmax = np.nanmax(valid_error_pixels_cropped) if valid_error_pixels_cropped.size > 0 else 1
        if error_vmin == error_vmax: error_vmax += 1e-6

        im3 = axs[i, 3].imshow(error_cropped_nan, cmap=cmap_error, vmin=error_vmin, vmax=error_vmax)
        axs[i, 3].set_title("Error (Crop)")
        fig.colorbar(im3, ax=axs[i, 3])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Update filename to reflect cropping
    if prefix == "Train":
        plot_filename = logging_parameters.model_output_folder_train / f"output_step_{current_step}_{prefix.lower()}_cropped.png"
    else:
        plot_filename = logging_parameters.model_output_folder_test / f"output_step_{current_step}_{prefix.lower()}_cropped.png"
    try:
        fig.savefig(plot_filename)
        logging.info(f"Saved cropped plot: {plot_filename}")
    except Exception as e:
        logging.error(f"Failed to save plot {plot_filename}: {e}")
    finally:
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
    ax.plot(steps, logging_parameters.loss_value, label="Train Loss", marker='o')
    # Ensure test loss aligns with the same steps
    ax.plot(steps, logging_parameters.test_loss_value[:len(steps)], label="Test Loss", marker='x') # Make sure lists have same length for plotting
    ax.set_xlabel("Training Step (Batch Index)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Training Step")
    ax.legend()
    ax.grid(True)

    # Ensure figure folder exists
    # logging_parameters.ensure_directories_exist() # Or handle Path object directly
    loss_filename = logging_parameters.figure_folder / "loss_curve.png"
    fig.savefig(loss_filename)
    plt.close(fig)
    logging.info(f"Saved loss curve: {loss_filename}")


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
            target_mask = batch['target_mask'].to(device)

            with autocast(device.type, enabled=use_amp):
                output_batch = model(pseudosection_batch, target_batch)
                loss_per_element = criterion(output_batch, target_batch) # Assumes criterion returns per-element loss

            # Apply mask and calculate batch loss
            masked_loss = loss_per_element * target_mask
            batch_loss_sum = masked_loss.sum()
            batch_elements = target_mask.sum()

            if batch_elements > 0:
                total_loss += batch_loss_sum.item()
                total_elements += batch_elements.item()

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
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool
) -> float:
    """
    Processes a single batch for training (forward, backward, optimization).

    Args:
        batch: A dictionary containing the batch data (tensors already on the correct device).
        model: The model to train.
        criterion: The loss function (expecting reduction='none').
        optimizer: The optimizer.
        scaler: GradScaler for mixed precision.
        device: The device being used.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        The calculated loss for this batch.
    """
    pseudosection_batch = batch['pseudosection'] # Already on device
    target_batch = batch['norm_log_resistivity_model'] # Already on device
    target_mask = batch['target_mask'] # Already on device

    optimizer.zero_grad()

    with autocast(device.type, enabled=use_amp):
        output_batch = model(pseudosection_batch, target_batch)
        loss_per_element = criterion(output_batch, target_batch) # Assumes reduction='none'

        # Apply mask and calculate effective batch loss
        masked_loss = loss_per_element * target_mask
        num_valid_elements = target_mask.sum()

        if num_valid_elements > 0:
            batch_loss = masked_loss.sum() / num_valid_elements
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

    return batch_loss.item()


def process_epoch(
    model: DynamicModel,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: Module,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    logging_parameters: LoggingParameters
) -> None:
    """
    Processes a single training epoch, including periodic evaluation and logging.

    Args:
        model: The model being trained.
        train_dataloader: DataLoader for training data.
        test_dataloader: DataLoader for test data.
        optimizer: The optimizer.
        criterion: The loss function.
        scaler: GradScaler for mixed precision.
        device: The device to train on.
        use_amp: Whether to use automatic mixed precision.
        epoch: The current epoch number (1-based).
        logging_parameters: Object for storing and managing logs.
    """
    model.train() # Ensure model is in training mode
    running_loss = 0.0
    num_batches_processed = 0

    # Prepare for potential plotting by getting one persistent test batch
    try:
      persistent_test_batch = next(iter(test_dataloader))
      # Move test batch to device once
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
            batch, model, criterion, optimizer, scaler, device, use_amp
        )
        running_loss += batch_loss
        num_batches_processed += 1

        # --- Logging and Evaluation ---
        if batch_idx in logging_parameters.print_points:
            current_step = (epoch - 1) * len(train_dataloader) + batch_idx # Global step
            # 1. Calculate Average Test Loss over the *entire* test set
            avg_test_loss = evaluate(model, test_dataloader, criterion, device, use_amp)

            # 2. Log losses
            logging_parameters.loss_value.append(batch_loss) # Log instantaneous train loss
            logging_parameters.test_loss_value.append(avg_test_loss)
            # Keep track of which batch index this corresponds to
            logging_parameters.print_points_list.append(current_step) # Log global step

            print(
                 f"\nEpoch {epoch} | Batch {batch_idx}/{len(train_dataloader)} | Step {current_step} | "
                 f"Train Loss (batch): {batch_loss:.5f} | "
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
                mask_test_np = tensors_to_numpy_list(persistent_test_batch['target_mask'].cpu())
                output_test_np = tensors_to_numpy_list(test_outputs.cpu())

                test_num_electrodes_list = persistent_test_batch['num_electrode'].cpu().numpy()
                test_array_type_list = [torch.argmax(array_type).cpu().numpy() for array_type in persistent_test_batch['array_type']]

                # Also plot the current training batch examples
                psec_train_np = tensors_to_numpy_list(batch['pseudosection'].cpu())
                target_train_np = tensors_to_numpy_list(batch['norm_log_resistivity_model'].cpu())
                mask_train_np = tensors_to_numpy_list(batch['target_mask'].cpu())

                train_num_electrodes_list = batch['num_electrode'].cpu().numpy()
                train_array_type_list = [torch.argmax(array_type).cpu().numpy() for array_type in batch['array_type']]
                # Need to recompute output for the training batch if not stored
                # Or get it from process_batch if refactored to return it
                # For simplicity, let's recompute (might be slightly different due to dropout state etc.)
                model.eval()
                with torch.no_grad():
                    with autocast(device.type, enabled=use_amp):
                        train_outputs_plot = model(batch['pseudosection'], batch['norm_log_resistivity_model'])
                model.train()
                output_train_np = tensors_to_numpy_list(train_outputs_plot.cpu())


                plot_samples(
                    psec_test_np, target_test_np, output_test_np, mask_test_np, test_num_electrodes_list, test_array_type_list, # Pass test mask
                    "Test", current_step, logging_parameters
                )
                plot_samples(
                    psec_train_np, target_train_np, output_train_np, mask_train_np, train_num_electrodes_list, train_array_type_list, # Pass train mask
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
    criterion: Module, # Make sure criterion has reduction='none'
    device: torch.device,
    use_amp: bool = True, # Flag to enable/disable AMP
    logging_parameters: LoggingParameters = None
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
    model.to(device)
    print(f"Using device: {device}")

    # AMP setup
    amp_enabled = use_amp and (device.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled)
    print(f"Automatic Mixed Precision (AMP): {'Enabled' if amp_enabled else 'Disabled'}")

    # --- Training Loop ---
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1): # Start epochs from 1
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")

        process_epoch(
            model,
            train_dataloader,
            test_dataloader,
            optimizer,
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
        print(f"Final Train Loss (last logged): {logging_parameters.loss_value[-1]:.5f}")
        print(f"Final Test Loss (last avg): {logging_parameters.test_loss_value[-1]:.5f}")
    else:
        print("No loss values were logged during training.")

    # Final loss curve plot
    plot_loss_curve(logging_parameters)
    print(f"Final loss curve saved to {logging_parameters.figure_folder}")
