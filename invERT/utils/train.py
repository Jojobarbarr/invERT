from time import perf_counter
start_time = perf_counter()
print("Importing PyTorch...")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
print(f"PyTorch imported in {perf_counter() - start_time:.2f} seconds")

from tqdm import tqdm
import matplotlib.pyplot as plt

from model.models import DynamicModel, init_weights
from data import IrregularDataset, custom_collate_fn, pre_process_data, denormalize

def initialize_dataset(normalized_data: list[torch.Tensor], normalized_target: list[torch.Tensor], batch_size: int) -> tuple[DataLoader, DataLoader]:
    print(f"Initializing dataset, dataloader and models...")
    dataset = IrregularDataset(normalized_data, normalized_target)

    train_size = int(0.8 * len(dataset))
    test_size: int = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)

    return train_dataloader, test_dataloader

def initialize_dataset_val(normalized_data_val: list[torch.Tensor], normalized_target_val: list[torch.Tensor], batch_size: int) -> DataLoader:
    dataset_val = IrregularDataset(normalized_data_val, normalized_target_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)

    return dataloader_val


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    min_shape: int = 5
    max_shape: int = 100
    assert min_shape <= max_shape

    noise: float = 0

    dataset_size: int = 1000
    batch_size: int = 1
    initial_lr: float = 0.0004
    lr_decrease_factor: float = 0.25
    patience: int = 5
    num_epochs: int = 50

    normalized_data, normalized_target, min_data, max_data, min_target, max_target, normalized_data_val, normalized_target_val = pre_process_data(dataset_size, min_shape, max_shape, noise)
    train_dataloader, test_dataloader = initialize_dataset(normalized_data, normalized_target, batch_size)
    dataloader_val = initialize_dataset_val(normalized_data_val, normalized_target_val, batch_size)

    # Initialize the models
    input_dim: int = 2  # MLP input size
    hidden_dim: list[int] = [32]  # MLP hidden layers size
    assert len(hidden_dim) > 0
    kernel_sizes: list[int] = [3, 3, 3, 3]  # CNN kernel sizes
    num_kernels: list[int] = [32, 64, 32, 1]  # CNN number of kernels
    assert len(kernel_sizes) == len(num_kernels)
    assert num_kernels[-1] == 1

    model = DynamicModel(input_dim, hidden_dim, num_kernels, kernel_sizes, 1).to(device)
    # model.apply(init_weights)
    print(f"Model: {model}")
    print(f"Model paramerers number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Set up the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_decrease_factor, patience=patience)

    # Training loop
    print("Training started...")
    loss_list: list[float] = []
    test_loss_list: list[float] = []
    print_point: int = len(train_dataloader) // 10
    try:
        for epoch in range(num_epochs):
            for batch, (inputs, targets) in tqdm(enumerate(train_dataloader), desc="Batch progression", total=len(train_dataloader), unit="batch"):

                optimizer.zero_grad()  # Clear previous gradients

                batch_loss: nn.MSELoss = 0

                # Forward pass through the MLP to get the kernels
                for input, target in zip(inputs, targets):
                    input: torch.Tensor = input.to(device)
                    target: torch.Tensor = target.to(device).unsqueeze(1)
                    input_metadata: torch.Tensor = torch.tensor([input.shape[1] / max_shape, input.shape[2] / max_shape], dtype=torch.float32).to(device)
                    # input_metadata: torch.Tensor = torch.tensor([input.shape[1], input.shape[2]], dtype=torch.float32).to(device)
                    output: torch.Tensor = model(input_metadata.unsqueeze(0), input.unsqueeze(1))

                    # Compute the loss
                    loss: nn.MSELoss = criterion(output[:, :, 1:-1, 1:-1], target[:, :, 1:-1, 1:-1])
                    batch_loss += loss
                batch_loss /= len(inputs)
                batch_loss.backward()

                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Check if there are any None gradients
                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         print(f"Grad for {name}: None")
                
                optimizer.step()

                if print_point == 0:
                    print_point = 1
                elif batch % print_point == 0:
                    model.eval()
                    test_batch_loss: nn.MSELoss = 0
                    test_batch: list[list[torch.Tensor]] = next(iter(test_dataloader))
                    test_inputs, test_targets = test_batch
                    with torch.no_grad():
                        for test_input, test_target in zip(test_inputs, test_targets):
                            test_input: torch.Tensor = test_input.to(device)
                            test_target: torch.Tensor = test_target.to(device).unsqueeze(1)
                            test_input_metadata: torch.Tensor = torch.tensor([test_input.shape[1] / max_shape, test_input.shape[2] / max_shape], dtype=torch.float32).to(device)
                            # test_input_metadata: torch.Tensor = torch.tensor([test_input.shape[1], test_input.shape[2]], dtype=torch.float32).to(device)
                            test_output = model(test_input_metadata.unsqueeze(0), test_input.unsqueeze(1))

                            # Compute the loss
                            test_loss = criterion(test_output[:, :, 1:-1, 1:-1], test_target[:, :, 1:-1, 1:-1])
                            test_batch_loss += test_loss
                    test_batch_loss /= len(test_inputs)
                    loss_list.append(batch_loss.item())
                    test_loss_list.append(test_batch_loss.item())
            scheduler.step(test_batch_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], train loss: {batch_loss.item():.5f}, test loss: {test_batch_loss.item():.5f}, lr: {optimizer.param_groups[0]["lr"]}')
    except KeyboardInterrupt:
        pass

    print("Training complete!")
    plt.plot(loss_list[1:], label="Train loss")
    plt.plot(test_loss_list[1:], label="Test loss")
    plt.title("Loss during training")
    plt.legend()
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(loss_list[10:], label="Train loss")
    plt.plot(test_loss_list[10:], label="Test loss")
    plt.title("Loss during training (after warmup)")
    plt.legend()
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.show()

    test_input, target = next(iter(dataloader_val))
    test_input = test_input[0].to(device)
    target = target[0].to(device).unsqueeze(0)
    test_input_metadata = torch.tensor([test_input.shape[1] / max_shape, test_input.shape[2] / max_shape], dtype=torch.float32).to(device)
    # test_input_metadata = torch.tensor([test_input.shape[1], test_input.shape[2]], dtype=torch.float32).to(device)

    test_output = model(test_input_metadata.unsqueeze(0), test_input.unsqueeze(1))
    
    error_map = torch.abs(test_output - target)

    CROP = True
    if CROP:
        test_output = test_output[:, :, 1:-1, 1:-1]
        target = target[:, :, 1:-1, 1:-1]
        error_map = error_map[:, :, 1:-1, 1:-1]

    min_val: float = min(test_output.min(), target.min()).item()
    max_val: float = max(test_output.max(), target.max()).item()

    plt.figure(figsize=(15, 5))

    # Plot the output
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
    plt.imshow(test_output[0, 0].detach().cpu().numpy(), cmap='viridis', vmin=min_val, vmax=max_val)
    plt.colorbar()
    plt.title('Output')

    # Plot the target
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, second subplot
    plt.imshow(target[0, 0].detach().cpu().numpy(), cmap='viridis', vmin=min_val, vmax=max_val)
    plt.colorbar()
    plt.title('Target')

    # Plot the error map
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, third subplot
    plt.imshow(error_map[0, 0].detach().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Error Map')

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

    # Denormalize the output and target
    denormalized_output = denormalize(test_output, min_target, max_target)
    denormalized_target = denormalize(target, min_target, max_target)

    error_map = torch.abs(denormalized_output - denormalized_target) / denormalized_target

    if CROP:
        denormalized_output = denormalized_output[:, :, 1:-1, 1:-1]
        denormalized_target = denormalized_target[:, :, 1:-1, 1:-1]
        error_map = error_map[:, :, 1:-1, 1:-1]
        
    min_val: float = min(denormalized_output.min(), denormalized_target.min()).item()
    max_val: float = max(denormalized_output.max(), denormalized_target.max()).item()

    # Plot the denormalized output and target
    plt.figure(figsize=(15, 5))

    # Plot the output
    plt.subplot(1, 3, 1)  # 1 row, 2 columns, first subplot
    plt.imshow(denormalized_output[0, 0].detach().cpu().numpy(), cmap='viridis', vmin=min_val, vmax=max_val)
    plt.colorbar()
    plt.title('Output')

    # Plot the target
    plt.subplot(1, 3, 2)  # 1 row, 2 columns, second subplot
    plt.imshow(denormalized_target[0, 0].detach().cpu().numpy(), cmap='viridis', vmin=min_val, vmax=max_val)
    plt.colorbar()
    plt.title('Target')

    # Plot the error map
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, third subplot
    plt.imshow(error_map[0, 0].detach().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Error Map')

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

    last_losses = 20
    print(f"Mean last losses: {sum(loss_list[-last_losses:]) / last_losses}")
    print(f"Mean last test losses: {sum(test_loss_list[-last_losses:]) / last_losses}")
    print(f"Mean error map: {error_map.mean()}")
