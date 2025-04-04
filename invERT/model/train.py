from tqdm import tqdm
from torch import Tensor, float32, no_grad, tensor
from torch.utils.data import DataLoader
# from torch.nn.utils import clip_grad_norm_
from torch.nn import Module
from torch.optim import Optimizer
from model.models import DynamicModel
import numpy as np
import matplotlib.pyplot as plt
from data.data import denormalize, invERTbatch
import multiprocessing as mp
from pathlib import Path
from model.parameters_classes import LoggingParameters
import torch

def plot_test(pseudosections: list[Tensor],
              log_norm_resistivity_models: list[Tensor],
              outputs: list[Tensor],
              logging_parameters: LoggingParameters,
              ) -> None:
    print_points_x = logging_parameters.print_points_list[:len(logging_parameters.loss_arrays)]
    plt.plot(print_points_x, logging_parameters.loss_arrays, label="Train Loss")
    plt.plot(print_points_x, logging_parameters.test_loss_arrays, label="Test Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss vs Batch")
    plt.legend()
    plt.savefig(logging_parameters.figure_path)
    plt.close()

    num_rows = 3
    num_cols = 4

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    fig.suptitle("Pseudosection, output, target and squared error")
    for i in range(num_rows):
        im0 = axs[i, 0].imshow(pseudosections[i].cpu().numpy(), cmap="viridis")
        axs[i, 0].set_title("Pseudosection")
        fig.colorbar(im0, ax=axs[i, 0])

        im1 = axs[i, 1].imshow(outputs[i].cpu().numpy(), cmap="viridis")
        axs[i, 1].set_title("Output")
        fig.colorbar(im1, ax=axs[i, 1])

        im2 = axs[i, 2].imshow(log_norm_resistivity_models[i].cpu().numpy(), cmap="viridis")
        axs[i, 2].set_title("Target")
        fig.colorbar(im2, ax=axs[i, 2])

        im3 = axs[i, 3].imshow((outputs[i] - log_norm_resistivity_models[i]) ** 2, cmap="viridis")
        axs[i, 3].set_title("Squared Error")
        fig.colorbar(im3, ax=axs[i, 3])
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(logging_parameters.model_output_path / f"output_{print_points_x[-1]}.png")
    plt.close(fig)


def test_batch(testing_batch: invERTbatch,
               model: DynamicModel,
               criterion: Module,
               device: str,
               logging_parameters: LoggingParameters,
               ) -> None:
    model.eval()

    pseudosections: list[Tensor] = []
    log_norm_resistivity_models: list[Tensor] = []
    outputs: list[Tensor] = []
    with torch.no_grad():
        # Initialize the batch loss value
        test_batch_loss_value: Tensor = tensor(0, dtype=float32).to(device)

        for sample in zip(*testing_batch):
            # Get the inputs and targets
            num_electrode, subsection_length, array, pseudosection, log_norm_resistivity_model = sample

            mlp_inputs: Tensor = tensor(
                [num_electrode, subsection_length, array], dtype=float32
            ).to(device).unsqueeze(0).unsqueeze(0)

            cnn_input: Tensor = tensor(
                pseudosection, dtype=float32
            ).to(device).unsqueeze(0).unsqueeze(0)

            target: Tensor = tensor(
                log_norm_resistivity_model, dtype=float32
            ).to(device).unsqueeze(0).unsqueeze(0)

            output: Tensor = model(mlp_inputs, cnn_input, target)

            # Compute the loss
            test_batch_loss_value += criterion(output, target)

            pseudosections.append(pseudosection.squeeze().cpu())
            log_norm_resistivity_models.append(log_norm_resistivity_model.squeeze().cpu())
            outputs.append(output.squeeze().cpu())

        # Normalize the loss
        test_batch_loss_value /= logging_parameters.batch_size

        logging_parameters.test_loss_arrays.append(test_batch_loss_value.item())

        plot_test(pseudosections, log_norm_resistivity_models, outputs, logging_parameters)

    model.train()




def process_batch(batch: invERTbatch,
                  batch_idx: int,
                  model: DynamicModel,
                  criterion: Module,
                  optimizer: Optimizer,
                  test_dataloader: DataLoader,
                  device: str,
                  logging_parameters: LoggingParameters,
                  ) -> None:
    # Clear previous gradients
    optimizer.zero_grad()
    # Initialize the batch loss value
    batch_loss_value: Tensor = tensor(0, dtype=float32).to(device)

    for sample in zip(*batch):
        num_electrode, subsection_length, array, pseudosection, log_norm_resistivity_model = sample

        mlp_inputs: Tensor = tensor(
            [num_electrode, subsection_length, array], dtype=float32
        ).to(device).unsqueeze(0).unsqueeze(0)

        cnn_input: Tensor = tensor(
            pseudosection, dtype=float32
        ).to(device).unsqueeze(0).unsqueeze(0)

        target: Tensor = tensor(
            log_norm_resistivity_model, dtype=float32
        ).to(device).unsqueeze(0).unsqueeze(0)

        output: Tensor = model(mlp_inputs, cnn_input, target)

        # Compute the loss
        batch_loss_value += criterion(output, target)

    # Normalize the loss
    batch_loss_value /= logging_parameters.batch_size
    # Backward pass
    batch_loss_value.backward()

    # # Check if there are any None gradients
    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         print(f"Grad for {name}: None")

    # Update the weights
    optimizer.step()

    testing_batch = next(iter(test_dataloader))
    if batch_idx in logging_parameters.print_points:
        test_batch(testing_batch, model, criterion, device, logging_parameters)


def process_epoch(model: DynamicModel,
                  train_dataloader: DataLoader,
                  test_dataloader: DataLoader,
                  optimizer: Optimizer,
                  criterion: Module,
                  device: str,
                  logging_parameters: LoggingParameters,
                  ) -> None:
    for batch_idx, batch in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        unit="batch"
    ):
        process_batch(
            batch, 
            batch_idx,
            model,
            criterion,
            optimizer,
            test_dataloader,
            device,
            logging_parameters
        )


def train(num_epochs: int,
          model: DynamicModel,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optimizer: Optimizer,
          criterion: Module,
          device: str,
          logging_parameters: LoggingParameters
          ):
    model.train()
    for epoch in range(num_epochs):
        process_epoch(
            model,
            train_dataloader,
            test_dataloader,
            optimizer,
            criterion,
            device,
            logging_parameters
        )
        # Print the loss values
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], ",
            f"train loss: {logging_parameters.loss_value[-1]:.5f}, "
            f"test loss: {logging_parameters.test_loss_value[-1]:.5f}, "
            f"lr: {optimizer.param_groups[0]['lr']}")
