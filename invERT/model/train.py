import logging
from tqdm import tqdm
from torch import Tensor, float32, no_grad, tensor
from torch.utils.data import DataLoader
# from torch.nn.utils import clip_grad_norm_
from torch.nn import Module
from torch.optim import Optimizer
from model.models import DynamicModel
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data.data import denormalize


def train(
        model: DynamicModel,
        epochs: int,
        train_dataloaders: list[DataLoader],
        test_dataloaders: list[DataLoader],
        optimizer: Optimizer,
        criterion: Module,  # Loss function
        scheduler: Module,  # Learning rate scheduler
        input_max_shape: int,
        device: str,
        print_points: int,
        save_plot_on_time: bool,
        output_folder: Path
) -> tuple[list[float],
           list[float],
           DynamicModel]:
    nb_batches: int = len(train_dataloaders) * len(train_dataloaders[0])
    loss_array: list[float] = []
    test_loss_array: list[float] = []
    for epoch in range(epochs):
        nbr_steps_total: int = nb_batches * epoch
        print(f"nbr_steps_total: {nbr_steps_total}")
        for batch in tqdm(range(nb_batches)):
            for dataloader_idx, train_dataloader in enumerate(
                    train_dataloaders):
                # Clear previous gradients
                optimizer.zero_grad()
                batch_loss_value: Tensor = \
                    tensor([0], dtype=float32).to(device)

                inputs, targets = next(iter(train_dataloader))
                logging.debug(
                    f"Input shape: {inputs.shape}, "
                    f"target shape: {targets.shape}")
                inputs: Tensor = inputs.to(device)
                targets: Tensor = targets.to(device)
                logging.debug(
                    f"Input shape: {inputs.shape}, "
                    f"target shape: {targets.shape}")
                logging.debug(
                    f"Input device: {inputs.device}, "
                    f"target device: {targets.device}")

                inputs_metadata: Tensor = tensor(
                    [
                        inputs.shape[1] / input_max_shape,
                        inputs.shape[2] / input_max_shape],
                    dtype=float32).to(device)

                logging.debug(f"input_metadata shape: {inputs_metadata.shape}")
                logging.debug(
                    f"input_metadata.unsqueeze(0) shape: "
                    f"{inputs_metadata.unsqueeze(0).shape}")
                outputs: Tensor = model(inputs_metadata.unsqueeze(0), inputs)

                # Compute the loss
                # Avoid to evaluate borders artifacts
                loss_value: Tensor = criterion(outputs, targets)
                batch_loss_value += loss_value

            batch_loss_value /= (train_dataloader.batch_size
                                 * len(train_dataloaders))
            batch_loss_value.backward()

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Check if there are any None gradients
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(f"Grad for {name}: None")

            optimizer.step()

            if (batch + 1) % print_points == 0:  # Test loss evaluation
                model.eval()
                test_batch_loss_value: Tensor = tensor(
                    [0], dtype=float32).to(device)
                for test_dataloader in test_dataloaders:
                    test_inputs, test_targets = next(iter(test_dataloader))
                    with no_grad():
                        test_inputs: Tensor = test_inputs.to(device)
                        test_targets: Tensor = test_targets.to(device)
                        test_input_metadata: Tensor = tensor(
                            [
                                test_inputs.shape[1] / input_max_shape,
                                test_inputs.shape[2] / input_max_shape],
                            dtype=float32).to(device)
                        test_outputs = model(
                            test_input_metadata.unsqueeze(0),
                            test_inputs)

                        # Compute the loss
                        # Avoid to evaluate borders artifacts
                        test_loss = criterion(
                            test_outputs,
                            test_targets)
                        test_batch_loss_value += test_loss

                test_batch_loss_value /= (test_dataloader.batch_size
                                          * len(test_dataloaders))
                loss_array.append(batch_loss_value.item())
                test_loss_array.append(test_batch_loss_value.item())
                if save_plot_on_time:
                    plt.plot(loss_array, label="Train loss")
                    plt.plot(test_loss_array, label="Test loss")
                    plt.legend(["train", "test"])
                    plt.xlabel("Step")
                    plt.ylabel("Loss")
                    plt.title("Train and test loss")
                    plt.savefig(output_folder / (
                        f"loss_at_step_{nbr_steps_total + batch}.png"))
                    plt.close()

        scheduler.step(test_batch_loss_value)
        print(
            f"Epoch [{epoch + 1}/{epochs}], ",
            f"train loss: {batch_loss_value.item():.5f}, "
            f"test loss: {test_batch_loss_value.item():.5f}, "
            f"lr: {optimizer.param_groups[0]['lr']}")

    return loss_array, test_loss_array, model


def print_model_results(
    model_list: list[DynamicModel],
    val_dataloaders: DataLoader,
    device: str,
    max_input_shape: int,
    min_data: float,
    max_data: float,
    min_target: float,
    max_target: float
) -> None:
    model = model_list[0]
    with no_grad():
        model.eval()
        for val_dataloader in val_dataloaders:
            val_inputs, val_targets = next(iter(val_dataloader))

            val_inputs: Tensor = val_inputs.to(device)
            val_targets: Tensor = val_targets.to(device)
            val_input_metadata: Tensor = tensor(
                [
                    val_inputs.shape[1] / max_input_shape,
                    val_inputs.shape[2] / max_input_shape],
                dtype=float32).to(device)
            val_outputs: Tensor = model(
                val_input_metadata.unsqueeze(0),
                val_inputs)

        print(
            f"val_inputs shape: {val_inputs.shape}, "
            f"val_outputs shape: {val_outputs.shape}")
        # Denormalize the data
        val_targets: np.ndarray[float, float] = denormalize(
            val_targets,
            min_target, max_target)[12, 0].detach().cpu().numpy()
        val_outputs: np.ndarray[float, float] = denormalize(
            val_outputs,
            min_target, max_target)[12, 0].detach().cpu().numpy()

        print(
            f"target shape: {val_targets.shape}, "
            f"output shape: {val_outputs.shape}")
        error_map = np.abs(val_targets - val_outputs) / val_targets

        # Create subplots: 1 row, 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Normalize target and output for consistent scaling
        vmin, vmax = min(
            val_targets.min(), val_outputs.min()), max(
            val_targets.max(), val_outputs.max())

        # Plot the target image
        im0 = axes[0].imshow(val_targets, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].set_title("Target")
        axes[0].axis('on')
        cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical')
        cbar0.set_label('Value')

        # Plot the output image
        im1 = axes[1].imshow(val_outputs, cmap='gray', vmin=vmin, vmax=vmax)
        axes[1].set_title("Output")
        axes[1].axis('on')
        cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical')
        cbar1.set_label('Value')

        # Plot the error map
        im2 = axes[2].imshow(error_map, cmap='hot')
        axes[2].set_title("Error Map")
        axes[2].axis('on')
        cbar2 = fig.colorbar(im2, ax=axes[2], orientation='vertical')
        cbar2.set_label('Relative Error')

        # Adjust layout to fit everything neatly
        plt.tight_layout()
        plt.show()
