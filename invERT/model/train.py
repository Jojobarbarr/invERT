from tqdm import tqdm
from torch import Tensor, float32, no_grad, tensor
from torch.utils.data import DataLoader
# from torch.nn.utils import clip_grad_norm_
from torch.nn import Module
from torch.optim import Optimizer
from model.models import DynamicModel
import numpy as np
import matplotlib.pyplot as plt
from data.data import denormalize
import multiprocessing as mp


def test_mini_batch(model: DynamicModel,
                    test_dataloader: DataLoader,
                    device: str,
                    input_max_shape: int,
                    criterion: Module,
                    ) -> Tensor:
    # Get the inputs and targets
    test_inputs, test_targets = next(iter(test_dataloader))

    mini_batch_size: int = test_inputs.shape[0]
    channels: int = test_inputs.shape[1]
    with no_grad():
        # Send the inputs and targets to the device
        test_inputs: Tensor = test_inputs.to(device)
        test_targets: Tensor = test_targets.to(device)

        # Compute the input metadata and send it to the device
        test_inputs_metadata: Tensor = tensor(
            [
                [test_inputs.shape[2] / input_max_shape,
                 test_inputs.shape[3] / input_max_shape]
            ]
            * mini_batch_size * channels
        ).view(mini_batch_size, channels, 2).to(device)

        # Forward pass
        test_outputs = model(test_inputs_metadata, test_inputs)

        # Compute the loss
        return criterion(test_outputs, test_targets)


def test(model: DynamicModel,
         print_point: int,
         device: str,
         test_dataloaders: list[DataLoader],
         input_max_shape: int,
         criterion: Module,
         loss_arrays: np.ndarray[float],
         test_loss_arrays: np.ndarray[float],
         batch_loss_value: Tensor,
         repetition: int,
         queue: mp.Queue
         ):
    model.eval()
    test_batch_loss_value: Tensor = tensor(0, dtype=float32).to(device)
    for test_dataloader in test_dataloaders:
        test_batch_loss_value += test_mini_batch(model,
                                                 test_dataloader,
                                                 device,
                                                 input_max_shape,
                                                 criterion)

    # Log the losses values
    loss_arrays[repetition, print_point] = batch_loss_value.item()
    test_loss_arrays[repetition, print_point] = test_batch_loss_value.item()

    # Send the losses to the queue
    queue.put((batch_loss_value.item(),
               test_batch_loss_value.item(),
               repetition,
               print_point))


def process_mini_batch(model: DynamicModel,
                       train_dataloader: list[DataLoader],
                       device: str,
                       input_max_shape: int,
                       criterion: Module,
                       ) -> Tensor:
    inputs, targets = next(iter(train_dataloader))

    mini_batch_size: int = inputs.shape[0]
    channels: int = inputs.shape[1]

    # Send the inputs and targets to the device
    inputs: Tensor = inputs.to(device)
    targets: Tensor = targets.to(device)

    # Compute the input metadata and send it to the device
    # TODO: Check if this is correct
    inputs_metadata: Tensor = tensor(
        [
            [inputs.shape[2] / input_max_shape,
                inputs.shape[3] / input_max_shape]
        ]
        * mini_batch_size * channels
    ).view(mini_batch_size, channels, 2).to(device)

    # Forward pass
    outputs: Tensor = model(inputs_metadata, inputs)

    # Compute the mini_batch loss
    return criterion(outputs, targets)


def process_batch(model: DynamicModel,
                  batch: int,
                  train_dataloaders: list[DataLoader],
                  test_dataloaders: list[DataLoader],
                  optimizer: Optimizer,
                  criterion: Module,
                  input_max_shape: int,
                  device: str,
                  logging_params: dict,
                  ) -> None:
    # Clear previous gradients
    optimizer.zero_grad()
    # Initialize the batch loss value
    batch_loss_value: Tensor = tensor(0, dtype=float32).to(device)

    for train_dataloader in train_dataloaders:
        # Accumulate the loss value for the mini-batch
        batch_loss_value += process_mini_batch(model,
                                                train_dataloader,
                                                device,
                                                input_max_shape,
                                                criterion)

    # Backward pass
    batch_loss_value.backward()

    # Gradient clipping
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Check if there are any None gradients
    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         print(f"Grad for {name}: None")

    # Update the weights
    optimizer.step()
    print_points: int = logging_params["print_points"]
    nb_print_points: int = logging_params["nb_print_points"]
    epoch: int = logging_params["epoch"]
    if (batch + 1) % print_points == 0:  # Test loss evaluation
        print_point: int = (batch // print_points) \
            + nb_print_points * epoch
        test_batch_loss_value = test(model,
                                        batch_size,
                                        print_point,
                                        device,
                                        test_dataloaders,
                                        input_max_shape,
                                        criterion,
                                        loss_arrays,
                                        test_loss_arrays,
                                        batch_loss_value,
                                        repetition,
                                        queue)
    return test_batch_loss_value


def process_epoch(model: DynamicModel,
                  nb_batches: int,
                  scheduler: Module,
                  ) -> tuple[Tensor, Tensor]:
    for batch in tqdm(range(nb_batches)):
        batch_loss_value, test_batch_loss_value = \
            process_batch(model,
                          batch,
                          )

    # Update the learning rate if needed
    scheduler.step(test_batch_loss_value)

    return batch_loss_value, test_batch_loss_value

def train(model: DynamicModel,
          epochs: int,
          nb_batches: int,
          batch_size: int,
          train_dataloaders: list[DataLoader],
          test_dataloaders: list[DataLoader],
          optimizer: Optimizer,
          criterion: Module,  # Loss function
          scheduler: Module,  # Learning rate scheduler
          input_max_shape: int,
          device: str,
          print_points: int,
          nb_print_points: int,
          loss_arrays: np.ndarray[float],
          test_loss_arrays: np.ndarray[float],
          repetition: int,
          queue: mp.Queue,
          ) -> tuple[list[float], list[float], DynamicModel]:
    for epoch in range(epochs):
        batch_loss_value, test_batch_loss_value = process_epoch(model)
        # Print the loss values
        print(
            f"Epoch [{epoch + 1}/{epochs}], ",
            f"train loss: {batch_loss_value.item():.5f}, "
            f"test loss: {test_batch_loss_value.item():.5f}, "
            f"lr: {optimizer.param_groups[0]['lr']}")

    return model


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
