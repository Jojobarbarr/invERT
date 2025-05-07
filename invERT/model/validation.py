import logging
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from model.parameters_classes import LoggingParameters
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.amp import autocast

from tqdm import tqdm


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> float:
    """Calculates Mean Absolute Error manually using NumPy."""
    if y_true.size == 0 or y_pred.size == 0: return float('nan')
    if sample_weight is not None:
        return np.sum(sample_weight * np.abs(y_true - y_pred)) / np.sum(sample_weight)
    return np.mean(np.abs(y_true - y_pred)).item()

def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> float:
    """Calculates Mean Squared Error manually using NumPy."""
    if y_true.size == 0 or y_pred.size == 0: return float('nan')
    if sample_weight is not None:
        return np.sum(sample_weight * (y_true - y_pred)**2) / np.sum(sample_weight)
    return np.mean((y_true - y_pred)**2).item()

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> float:
    """Calculates Root Mean Squared Error manually using NumPy."""
    mse = calculate_mse(y_true, y_pred, sample_weight)
    return np.sqrt(mse) if not np.isnan(mse) else float('nan')

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> float:
    """Calculates R-squared (Coefficient of Determination) manually using NumPy."""
    if y_true.size == 0 or y_pred.size == 0: return float('nan')
    
    if sample_weight is not None:
        # Weighted R² calculation
        ss_res = np.sum(sample_weight * (y_true - y_pred)**2)
        ss_tot = np.sum(sample_weight * (y_true - np.mean(y_true))**2)
    else:
        # Unweighted R² calculation
        ss_res = np.sum((y_true - y_pred)**2) # Residual sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true))**2) # Total sum of squares
    
    if ss_tot == 0:
        # Handle case where true values are constant: R² is undefined or 1 if predictions match exactly
        return 1.0 if ss_res == 0 else 0.0 # Or return nan, depending on desired behavior
        # Returning 0 might be safer to indicate no variance explained
    
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()





def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, logging_params: LoggingParameters):
    """Plots residuals (y_pred - y_true) vs. true values."""
    if y_true.size == 0 or y_pred.size == 0:
        logging.warning("Cannot plot residuals: No data provided.")
        return
    print("Plotting final validation residuals...")
    residuals = y_pred - y_true
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, residuals, alpha=0.2)
        plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
        plt.xlabel("True Values")
        plt.ylabel("Residuals (Predicted - True)")
        plt.title("Final Validation: Residuals vs. True Values")
        plt.grid(True)
        plt.legend(loc='upper left')
        plot_path = logging_params.validation_folder / "final_validation_residuals.png"
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting residuals: {e}")

def plot_residual_histogram(y_true: np.ndarray, y_pred: np.ndarray, logging_params: LoggingParameters):
    """Plots a histogram of the residuals."""
    if y_true.size == 0 or y_pred.size == 0:
        logging.warning("Cannot plot residual histogram: No data provided.")
        return
    print("Plotting final validation residual histogram...")
    residuals = y_pred - y_true
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, density=True, alpha=0.7, label='Residual Distribution')
        # Optional: Fit a normal distribution
        # from scipy.stats import norm
        # mu, std = norm.fit(residuals)
        # xmin, xmax = plt.xlim()
        # x = np.linspace(xmin, xmax, 100)
        # p = norm.pdf(x, mu, std)
        # plt.plot(x, p, 'k', linewidth=2, label=f'Fit results: mu={mu:.2f}, std={std:.2f}')
        plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
        plt.xlabel("Residual Value (Predicted - True)")
        plt.ylabel("Density")
        plt.title("Final Validation: Histogram of Residuals")
        plt.legend(loc='upper left')
        plt.grid(True)
        plot_path = logging_params.validation_folder / "final_validation_residual_histogram.png"
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting residual histogram: {e}")


def plot_final_scatter(all_targets_np, all_predictions_np, logging_params):
    """Plots a scatter plot of true vs predicted values for final validation."""
    print("Plotting final validation scatter plot...")
    try:
        plt.figure(figsize=(8, 8))
        min_val = min(all_targets_np.min(), all_predictions_np.min())
        max_val = max(all_targets_np.max(), all_predictions_np.max())
        plt.scatter(all_targets_np.flatten(), all_predictions_np.flatten(), alpha=0.1)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal y=x')
        plt.xlabel("True Values (Flattened)")
        plt.ylabel("Predicted Values (Flattened)")
        plt.title("Final Validation: True vs. Predicted Values")
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plot_path = logging_params.validation_folder / "final_validation_scatter.png"
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting final scatter plot: {e}")

def plot_single_validation_sample(
    psec: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray, # Add weight matrix for context
    sample_idx: int,
    logging_params: LoggingParameters
):
    """Plots and saves a single validation sample comparison."""
    try:
        fig, axs = plt.subplots(1, 4, figsize=(20, 5)) # Added weight plot
        
        # Determine consistent color limits for target and prediction
        valid_target = target[weight > 1e-9] # Consider only weighted pixels for range
        valid_prediction = prediction[weight > 1e-9]
        if valid_target.size > 0 and valid_prediction.size > 0:
            vmin = min(np.min(valid_target), np.min(valid_prediction))
            vmax = max(np.max(valid_target), np.max(valid_prediction))
        else: # Fallback if no valid pixels
            vmin, vmax = np.min(target), np.max(target) 
            if math.isnan(vmin) or math.isinf(vmin) or math.isnan(vmax) or math.isinf(vmax):
                 vmin, vmax = 0, 1 # Safe fallback

        psec_np = psec.squeeze()[0]
        psec_np = np.where(psec_np != 0, psec_np, np.nan)
        # Plot Pseudosection (Input)
        im0 = axs[0].imshow(psec_np, cmap='viridis') # Squeeze removes channel dim if present
        axs[0].set_title("Pseudosection (Input)")
        fig.colorbar(im0, ax=axs[0])
        axs[0].axis('off')

        # Plot Target Resistivity
        im1 = axs[1].imshow(target.squeeze(), cmap='viridis', vmin=0, vmax=1)
        axs[1].set_title("Target Resistivity")
        fig.colorbar(im1, ax=axs[1])
        axs[1].axis('off')

        # Plot Predicted Resistivity
        im2 = axs[2].imshow(prediction.squeeze(), cmap='viridis', vmin=0, vmax=1)
        axs[2].set_title("Predicted Resistivity")
        fig.colorbar(im2, ax=axs[2])
        axs[2].axis('off')

        # Plot Weights Matrix
        im3 = axs[3].imshow(weight.squeeze(), cmap='magma', vmin=0) # Use binary_r, vmin=0
        axs[3].set_title("Weights (JtJ Diag)")
        fig.colorbar(im3, ax=axs[3])
        axs[3].axis('off')

        fig.suptitle(f"Final Validation Sample {sample_idx}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

        # Create subdirectory if it doesn't exist
        sample_plot_dir = logging_params.validation_folder / "final_validation_samples"
        sample_plot_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = sample_plot_dir / f"validation_sample_{sample_idx:04d}.png"
        plt.savefig(plot_path)
        plt.close(fig)
    except Exception as e:
        logging.error(f"Error plotting validation sample {sample_idx}: {e}")


def final_validation(
    model,
    dataloader: DataLoader,
    criterion: Module,
    device: torch.device,
    use_amp: bool,
    logging_parameters: LoggingParameters
) -> Dict[str, float]:
    """
    Performs final evaluation, calculates metrics manually, generates summary plots,
    and saves individual sample plots.

    Args:
        model: The trained model.
        dataloader: DataLoader for the final validation/test dataset.
        criterion: The loss function (reduction='none').
        device: The device to run evaluation on.
        use_amp: Whether to use automatic mixed precision.
        logging_parameters: For saving plots and logs.

    Returns:
        A dictionary containing final evaluation metrics.
    """
    print("\n--- Starting Final Validation ---")
    model.eval()  # Set model to evaluation mode

    # Lists to store *individual* sample data (after potential cropping in non-batch mode)
    all_inputs_list = []
    all_predictions_list = []
    all_targets_list = []
    all_weights_list = []

    total_loss = 0.0
    total_elements = 0.0 # Sum of weights

    with torch.no_grad(): # Disable gradient calculations
        sample_idx_counter = 0
        for batch in tqdm(dataloader, desc="Final Validation", leave=True, unit="batch"):
            # Keep a CPU copy for potential later use if needed, but move necessary parts to device
            # batch_cpu = {k: v.cpu() for k, v in batch.items()} # Less efficient if only used for plotting later
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            with autocast(device.type, enabled=use_amp):
                batch_loss_sum = 0.0
                current_batch_weights_sum = 0.0

                if hasattr(model, 'batch_processing') and model.batch_processing:
                    # --- Batch Processing Mode ---
                    pseudosection_batch = batch_gpu['pseudosection']
                    target_batch = batch_gpu['norm_log_resistivity_model']
                    weights_matrix = batch_gpu['JtJ_diag']

                    output_batch = model(pseudosection_batch, target_batch) # Target maybe needed
                    weighted_loss = weights_matrix * criterion(output_batch, target_batch)

                    # Iterate through samples *within* the batch for storing individual data
                    for i in range(pseudosection_batch.size(0)):
                        psec_sample = pseudosection_batch[i]
                        target_sample = target_batch[i]
                        output_sample = output_batch[i]
                        weight_sample = weights_matrix[i]

                        sample_weights_sum = weight_sample.sum().item()
                        if sample_weights_sum > 0:
                            sample_loss_sum = (weight_sample * criterion(output_sample.unsqueeze(0), target_sample.unsqueeze(0))).sum().item()
                            
                            batch_loss_sum += sample_loss_sum
                            current_batch_weights_sum += sample_weights_sum

                            # Store individual sample data (moved to CPU)
                            all_inputs_list.append(psec_sample.cpu())
                            all_targets_list.append(target_sample.cpu())
                            all_predictions_list.append(output_sample.cpu())
                            all_weights_list.append(weight_sample.cpu())
                        else:
                            # Optionally store samples even if weights are zero, for plotting
                            # all_inputs_list.append(psec_sample.cpu())
                            # all_targets_list.append(target_sample.cpu())
                            # all_predictions_list.append(output_sample.cpu()) # Prediction might be weird
                            # all_weights_list.append(weight_sample.cpu())
                            logging.debug(f"Skipping sample {sample_idx_counter + i} loss calculation due to zero weights.")
                            continue # Skip loss accumulation if weights sum to 0

                else:
                    # --- Sample-by-Sample Processing Mode ---
                    pseudosection_batch = batch_gpu['pseudosection']
                    target_batch = batch_gpu['norm_log_resistivity_model']
                    weights_matrix = batch_gpu['JtJ_diag']
                    pseudo_masks = batch_gpu['pseudo_masks']
                    target_masks = batch_gpu['target_masks']

                    for pseudo, target, weight_mat, p_mask, t_mask in zip(
                        pseudosection_batch, target_batch, weights_matrix, pseudo_masks, target_masks
                    ):
                        ph, pw = p_mask.tolist()
                        th, tw = t_mask.tolist()

                        # Crop inputs and targets for this specific sample
                        pseudo_crop = pseudo[..., :ph, :pw].unsqueeze(0)
                        target_crop = target[..., :th, :tw].unsqueeze(0)
                        weight_mat_crop = weight_mat[..., :th, :tw].unsqueeze(0)

                        output = model(pseudo_crop, target_crop) # Target maybe needed
                        weighted_loss = weight_mat_crop * criterion(output, target_crop)

                        sample_loss_sum = weighted_loss.sum().item()
                        sample_weights_sum = weight_mat_crop.sum().item()

                        if sample_weights_sum > 0:
                            batch_loss_sum += sample_loss_sum
                            current_batch_weights_sum += sample_weights_sum

                            # Store CROPPED data (moved to CPU)
                            all_inputs_list.append(pseudo_crop.squeeze(0).cpu()) # Remove batch dim
                            all_targets_list.append(target_crop.squeeze(0).cpu())
                            all_predictions_list.append(output.squeeze(0).cpu())
                            all_weights_list.append(weight_mat_crop.squeeze(0).cpu())
                        else:
                            # Optionally store samples even if weights are zero
                            # all_inputs_list.append(pseudo_crop.squeeze(0).cpu())
                            # all_targets_list.append(target_crop.squeeze(0).cpu())
                            # all_predictions_list.append(output.squeeze(0).cpu())
                            # all_weights_list.append(weight_mat_crop.squeeze(0).cpu())
                            logging.debug(f"Skipping sample loss calculation due to zero weights.")
                            continue # Skip loss accumulation

            total_loss += batch_loss_sum
            total_elements += current_batch_weights_sum
            sample_idx_counter += len(batch['pseudosection']) # Increment based on actual batch size processed


    # --- Aggregate Results for Metrics (Handles Variable Shapes) ---
    if not all_targets_list: # Check if any samples were stored
        logging.error("No samples collected during final validation.")
        # Attempt to calculate loss if elements > 0, otherwise return inf
        final_avg_loss = total_loss / total_elements if total_elements > 0 else float('inf')
        return {"final_avg_loss": final_avg_loss}

    # Calculate final average loss
    final_avg_loss = total_loss / total_elements if total_elements > 0 else float('inf')
    metrics = {"final_avg_loss": final_avg_loss}

    print("Aggregating results for metrics across all samples (handling variable shapes)...")
    flat_targets_list = []
    flat_predictions_list = []
    flat_weights_list = []

    # Iterate through collected *individual* samples (which are CPU tensors)
    for target_i, pred_i, weight_i in tqdm(zip(all_targets_list, all_predictions_list, all_weights_list),
                                           total=len(all_targets_list), desc="Flattening/Masking Samples"):
        # Create mask for the current sample
        # Ensure weight_i has compatible dimensions for masking target_i and pred_i
        # If weight is [H, W] and target/pred are [C, H, W], unsqueeze weight
        if weight_i.dim() < target_i.dim():
             weight_mask = weight_i.unsqueeze(0).expand_as(target_i) # Expand channels if needed
        else:
             weight_mask = weight_i
        
        valid_mask_i = weight_mask > 0.3 # Use threshold comparison

        # Select valid elements using the mask
        # These will be 1D tensors (or empty tensors if no valid points)
        target_flat_i = torch.masked_select(target_i, valid_mask_i)
        pred_flat_i = torch.masked_select(pred_i, valid_mask_i)
        weight_flat_i = torch.masked_select(weight_mask, valid_mask_i) # Mask weights too

        # Append the 1D tensor of valid elements
        flat_targets_list.append(target_flat_i)
        flat_predictions_list.append(pred_flat_i)
        flat_weights_list.append(weight_flat_i)

    # Concatenate the lists of 1D tensors into single large 1D tensors
    if flat_targets_list: # Check if any valid data was collected after masking
        try:
            targets_flat_all = torch.cat(flat_targets_list, dim=0)
            predictions_flat_all = torch.cat(flat_predictions_list, dim=0)
            weights_flat_all = torch.cat(flat_weights_list, dim=0)

            # Convert to NumPy for metric calculation and plotting
            targets_np = targets_flat_all.numpy()
            predictions_np = predictions_flat_all.numpy()
            weights_np = weights_flat_all.numpy()


            print(f"Calculating metrics on {targets_np.size} valid data points across all samples.")
            # Calculate metrics using manual functions
            metrics["final_mae"] = calculate_mae(targets_np, predictions_np, weights_np)
            metrics["final_mse"] = calculate_mse(targets_np, predictions_np, weights_np)
            metrics["final_rmse"] = calculate_rmse(targets_np, predictions_np, weights_np)
            metrics["final_r2"] = calculate_r2(targets_np, predictions_np, weights_np)

        except Exception as e:
            logging.error(f"Error during metric calculation or conversion: {e}")
            targets_np = np.array([]) # Ensure empty array for plotting checks
            predictions_np = np.array([])
            weights_np = np.array([]) # Ensure empty array for plotting checks
            # Add NaN placeholders
            metrics["final_mae"] = float('nan')
            metrics["final_mse"] = float('nan')
            metrics["final_rmse"] = float('nan')
            metrics["final_r2"] = float('nan')
    else:
        print("No valid data points found after masking samples with weights. Skipping MAE, MSE, RMSE, R2.")
        targets_np = np.array([]) # Ensure empty array for plotting checks
        predictions_np = np.array([])
        weights_np = np.array([]) # Ensure empty array for plotting checks
        metrics["final_mae"] = float('nan')
        metrics["final_mse"] = float('nan')
        metrics["final_rmse"] = float('nan')
        metrics["final_r2"] = float('nan')

    print("\n--- Final Validation Metrics (Manual) ---")
    for name, value in metrics.items():
        if not math.isnan(value):
             print(f"{name}: {value:.6f}")
        else:
             print(f"{name}: NaN")

    # --- Generate Summary Plots ---
    if targets_np.size > 0: # Check using the final aggregated NumPy array
        targets_np_weight = targets_np[weights_np > 0.1]
        predictions_np_weight = predictions_np[weights_np > 0.1]
        plot_final_scatter(targets_np_weight, predictions_np_weight, logging_parameters)
        plot_residuals(targets_np_weight, predictions_np_weight, logging_parameters)
        plot_residual_histogram(targets_np_weight, predictions_np_weight, logging_parameters)
    else:
        print("Skipping summary plots due to no valid (weighted) data points.")

    # --- Save Individual Sample Plots ---
    print(f"Saving individual plots for {len(all_inputs_list)} collected validation samples...")
    # The rest of this section remains the same, operating on the stored lists
    for i in range(len(all_inputs_list)):
        try:
            # Convert tensors to numpy just before plotting
            psec_np = all_inputs_list[i].numpy()
            target_np = all_targets_list[i].numpy()
            pred_np = all_predictions_list[i].numpy()
            weight_np = all_weights_list[i].numpy()

            plot_single_validation_sample(
                psec_np, target_np, pred_np, weight_np, i, logging_parameters
            )
            if (i + 1) % 100 == 0:
                print(f"  Saved {i+1}/{len(all_inputs_list)} sample plots...")
        except Exception as e:
            logging.error(f"Failed to plot or save sample {i}: {e}")
            # Continue to next sample

    print("--- Final Validation Finished ---")
    model.train() # Set back to train mode
    return metrics
