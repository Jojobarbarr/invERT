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
              test_pseudosections: list[Tensor],
              test_log_norm_resistivity_models: list[Tensor],
              test_outputs: list[Tensor],
              logging_parameters: LoggingParameters,
              ) -> None:
    print_points_x = logging_parameters.print_points_list[:len(logging_parameters.loss_value)]
    plt.plot(print_points_x, logging_parameters.loss_value, label="Train Loss")
    plt.plot(print_points_x, logging_parameters.test_loss_value, label="Test Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss vs Batch")
    plt.legend()
    plt.savefig(logging_parameters.figure_folder / "loss.png")
    plt.close()

    num_rows = 3
    num_cols = 5

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    fig.suptitle("Pseudosection, output, target, squared error and weighted squared error")
    for i in range(num_rows):
        vmin = min(
            np.min(test_outputs[i]),
            np.min(test_log_norm_resistivity_models[i]),
        )
        vmax = max(
            np.max(test_outputs[i]),
            np.max(test_log_norm_resistivity_models[i]),
        )
        im0 = axs[i, 0].imshow(test_pseudosections[i], cmap="viridis")
        axs[i, 0].set_title("Pseudosection")
        fig.colorbar(im0, ax=axs[i, 0])

        im1 = axs[i, 1].imshow(test_outputs[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axs[i, 1].set_title("Output")
        fig.colorbar(im1, ax=axs[i, 1])

        im2 = axs[i, 2].imshow(test_log_norm_resistivity_models[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axs[i, 2].set_title("Target")
        fig.colorbar(im2, ax=axs[i, 2])
        
        squared_error = (test_outputs[i] - test_log_norm_resistivity_models[i]) ** 2

        im3 = axs[i, 3].imshow(squared_error, cmap="viridis")
        axs[i, 3].set_title("Squared Error")
        fig.colorbar(im3, ax=axs[i, 3])

        h, w = squared_error.shape
        weights = 10 * np.linspace(1, 0.1, num=h).reshape(h, 1)
        weights = np.repeat(weights, w, axis=1)  # shape (h, w)
        weighted_squared_error = squared_error * weights
        im4 = axs[i, 4].imshow(weighted_squared_error, cmap="viridis")
        axs[i, 4].set_title("Weighted Squared Error")
        fig.colorbar(im4, ax=axs[i, 4])

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(logging_parameters.model_output_folder / f"output_{print_points_x[-1]}_test.png")
    plt.close(fig)


    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    fig.suptitle("Pseudosection, output, target, squared error and weighted squared error")
    for i in range(num_rows):
        vmin = min(
            np.min(outputs[i]),
            np.min(log_norm_resistivity_models[i]),
        )
        vmax = max(
            np.max(outputs[i]),
            np.max(log_norm_resistivity_models[i]),
        )
        im0 = axs[i, 0].imshow(pseudosections[i], cmap="viridis")
        axs[i, 0].set_title("Pseudosection")
        fig.colorbar(im0, ax=axs[i, 0])

        im1 = axs[i, 1].imshow(outputs[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axs[i, 1].set_title("Output")
        fig.colorbar(im1, ax=axs[i, 1])

        im2 = axs[i, 2].imshow(log_norm_resistivity_models[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axs[i, 2].set_title("Target")
        fig.colorbar(im2, ax=axs[i, 2])
        
        squared_error = (outputs[i] - log_norm_resistivity_models[i]) ** 2

        im3 = axs[i, 3].imshow(squared_error, cmap="viridis")
        axs[i, 3].set_title("Squared Error")
        fig.colorbar(im3, ax=axs[i, 3])

        h, w = squared_error.shape
        weights = 10 * np.linspace(1, 0.1, num=h).reshape(h, 1)
        weights = np.repeat(weights, w, axis=1)  # shape (h, w)
        weighted_squared_error = squared_error * weights
        im4 = axs[i, 4].imshow(weighted_squared_error, cmap="viridis")
        axs[i, 4].set_title("Weighted Squared Error")
        fig.colorbar(im4, ax=axs[i, 4])

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(logging_parameters.model_output_folder / f"output_{print_points_x[-1]}_train.png")
    plt.close(fig)


def test_batch(testing_batch: invERTbatch,
               model: DynamicModel,
               criterion: Module,
               device: str,
               pseudosections: list[Tensor],
               log_norm_resistivity_models: list[Tensor],
               outputs: list[Tensor],
               logging_parameters: LoggingParameters,
               ) -> None:
    model.eval()

    test_pseudosections: list[Tensor] = []
    test_log_norm_resistivity_models: list[Tensor] = []
    test_outputs: list[Tensor] = []
    with torch.no_grad():
        # Initialize the batch loss value
        test_batch_loss_value: Tensor = tensor(0, dtype=float32).to(device)

        for sample in zip(*testing_batch):
            # Get the inputs and targets
            num_electrode, subsection_length, array, pseudosection, log_norm_resistivity_model = sample

            # mlp_inputs: Tensor = tensor(
            #     [num_electrode, subsection_length, array], dtype=float32
            # ).to(device).unsqueeze(0)

            cnn_input: Tensor = tensor(
                pseudosection, dtype=float32
            ).to(device).unsqueeze(0).unsqueeze(0)

            target: Tensor = tensor(
                log_norm_resistivity_model, dtype=float32
            ).to(device).unsqueeze(0).unsqueeze(0)

            # output: Tensor = model(mlp_inputs, cnn_input, target)
            output: Tensor = model(cnn_input, target)

            # input_flat: Tensor = tensor(
            #     pseudosection, dtype=float32
            # ).to(device).view(-1)

            # target: Tensor = tensor(
            #     log_norm_resistivity_model, dtype=float32
            # ).to(device)

            # output = model(input_flat, target)

            # h, w = output.shape[2], output.shape[3]
            # weights = 10 * torch.linspace(1, 0.1, steps=h, device=device).view(1, 1, h, 1).expand(1, 1, h, w)
            # Compute the loss
            # test_batch_loss_value += criterion(output * weights, target * weights) 
            test_batch_loss_value += criterion(output, target)

            test_pseudosections.append(pseudosection)
            test_log_norm_resistivity_models.append(log_norm_resistivity_model)
            test_outputs.append(output.squeeze().cpu().numpy())

        # Normalize the loss
        test_batch_loss_value /= logging_parameters.batch_size

        logging_parameters.test_loss_value.append(test_batch_loss_value.item())

        plot_test(pseudosections, log_norm_resistivity_models, outputs, test_pseudosections, test_log_norm_resistivity_models, test_outputs, logging_parameters)

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

    pseudosections: list[Tensor] = []
    log_norm_resistivity_models: list[Tensor] = []
    outputs: list[Tensor] = []

    for sample in zip(*batch):
        num_electrode, subsection_length, array, pseudosection, log_norm_resistivity_model = sample

        # mlp_inputs: Tensor = tensor(
        #     [num_electrode, subsection_length, array], dtype=float32
        # ).to(device).unsqueeze(0)

        cnn_input: Tensor = tensor(
            pseudosection, dtype=float32
        ).to(device).unsqueeze(0).unsqueeze(0)

        target: Tensor = tensor(
            log_norm_resistivity_model, dtype=float32
        ).to(device).unsqueeze(0).unsqueeze(0)

        # output: Tensor = model(mlp_inputs, cnn_input, target)
        output: Tensor = model(cnn_input, target)
        
        # input_flat: Tensor = tensor(
        #     pseudosection, dtype=float32
        # ).to(device).view(-1)

        # target: Tensor = tensor(
        #     log_norm_resistivity_model, dtype=float32
        # ).to(device)

        # output = model(input_flat, target)

        # Compute the loss
        # h, w = output.shape[2], output.shape[3]
        # weights = 10 * torch.exp(-torch.linspace(0, 4, steps=h, device=device)).view(1, 1, h, 1).expand(1, 1, h, w)
        # batch_loss_value += criterion(output * weights, target * weights)
        batch_loss_value += criterion(output, target)
        if batch_idx in logging_parameters.print_points:
            outputs.append(output.squeeze().detach().cpu().numpy())
            pseudosections.append(pseudosection)
            log_norm_resistivity_models.append(log_norm_resistivity_model)

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

    if batch_idx in logging_parameters.print_points:
        testing_batch = next(iter(test_dataloader))
        logging_parameters.loss_value.append(batch_loss_value.item())
        test_batch(testing_batch, model, criterion, device, pseudosections, log_norm_resistivity_models, outputs, logging_parameters)


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
