import logging
from tqdm import tqdm
from torch import Tensor, float32, no_grad, tensor
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn import Module
from torch.optim import Optimizer
from model.models import DynamicModel
import numpy as np
import matplotlib.pyplot as plt
from data.data import denormalize

def train(
        model: DynamicModel,
        epochs: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: Optimizer,
        criterion: Module,  # Loss function
        scheduler: Module, # Learning rate scheduler
        loss_array: np.ndarray[float],
        test_loss_array: np.ndarray[float],
        input_max_shape: int,
        device: str,
        print_points: int,
) -> None:
    for epoch in range(epochs):
        for batch, (inputs, targets) in tqdm(enumerate(train_dataloader), desc="Batch progression", total=len(train_dataloader), unit="batch"):

            optimizer.zero_grad()  # Clear previous gradients

            batch_loss_value: Tensor = tensor([0], dtype=float32).to(device)

            for input, target in zip(inputs, targets):
                logging.debug(f"Input shape: {input.shape}, target shape: {target.shape}")
                input: Tensor = input.to(device).unsqueeze(1)
                target: Tensor = target.to(device).unsqueeze(1)
                logging.debug(f"Input shape: {input.shape}, target shape: {target.shape}")
                logging.debug(f"Input device: {input.device}, target device: {target.device}")
                
                input_metadata: Tensor = tensor([input.shape[1] / input_max_shape, input.shape[2] / input_max_shape], dtype=float32).to(device)
                
                logging.debug(f"input_metadata shape: {input_metadata.shape}")
                logging.debug(f"input_metadata.unsqueeze(0) shape: {input_metadata.unsqueeze(0).shape}")
                output: Tensor = model(input_metadata.unsqueeze(0), input)

                # Compute the loss
                loss_value: Tensor = criterion(output[:, :, 1:-1, 1:-1], target[:, :, 1:-1, 1:-1]) # Avoid to evaluate borders artifacts
                batch_loss_value += loss_value

            batch_loss_value /= len(inputs)
            batch_loss_value.backward()

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Check if there are any None gradients
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(f"Grad for {name}: None")
            
            optimizer.step()

            if (batch + 1) % print_points == 0: # Test loss evaluation
                model.eval()
                test_batch_loss_value: Tensor = tensor([0], dtype=float32).to(device)
                test_batch: list[list[Tensor]] = next(iter(test_dataloader))
                test_inputs, test_targets = test_batch
                with no_grad():
                    for test_input, test_target in zip(test_inputs, test_targets):
                        test_input: Tensor = test_input.to(device)
                        test_target: Tensor = test_target.to(device).unsqueeze(1)
                        test_input_metadata: Tensor = tensor([test_input.shape[1] / input_max_shape, test_input.shape[2] / input_max_shape], dtype=float32).to(device)
                        test_output = model(test_input_metadata.unsqueeze(0), test_input.unsqueeze(1))

                        # Compute the loss
                        test_loss = criterion(test_output[:, :, 1:-1, 1:-1], test_target[:, :, 1:-1, 1:-1]) # Avoid to evaluate borders artifacts
                        test_batch_loss_value += test_loss

                test_batch_loss_value /= len(test_inputs)

                loss_array[batch // print_points] = batch_loss_value.item()
                test_loss_array[batch // print_points] = test_batch_loss_value.item()

        scheduler.step(test_batch_loss_value)
        print(f'Epoch [{epoch + 1}/{epochs}], train loss: {batch_loss_value.item():.5f}, test loss: {test_batch_loss_value.item():.5f}, lr: {optimizer.param_groups[0]["lr"]}')
    
    return loss_array, test_loss_array, model

def print_model_results(model_list: list[DynamicModel], val_dataloader: DataLoader, device: str, max_input_shape: int, min_data: float, max_data: float, min_target: float, max_target: float) -> None:
    model = model_list[0]
    with no_grad():
        model.eval()
        val_batch: list[list[Tensor]] = next(iter(val_dataloader))
        val_inputs, val_targets = val_batch
        for val_input, val_target in zip(val_inputs, val_targets):
            val_input: Tensor = val_input.to(device)
            val_target: Tensor = val_target.to(device).unsqueeze(1)
            val_input_metadata: Tensor = tensor([val_input.shape[1] / max_input_shape, val_input.shape[2] / max_input_shape], dtype=float32).to(device)
            val_output = model(val_input_metadata.unsqueeze(0), val_input.unsqueeze(1))

            # Denormalize the data
            val_input = denormalize(val_input, min_data, max_data)
            val_target = denormalize(val_target, min_target, max_target)
            val_output = denormalize(val_output[:, :, 1:-1, 1:-1], min_target, max_target)

            # Plot the data
            plt.imshow(val_input[0, 0].detach().cpu().numpy(), cmap='gray')
            plt.show()
            plt.imshow(val_target[0, 0].detach().cpu().numpy(), cmap='gray')
            plt.show()
            plt.imshow(val_output[0, 0].detach().cpu().numpy(), cmap='gray')
            plt.show()