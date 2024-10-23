print("Importing PyTorch...")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import matplotlib.pyplot as plt

from models import KernelGeneratorMLP, DynamicConvNet


# Assuming you have a dataset class defined
class MyDataset(Dataset):
    def __init__(self, data: list[torch.tensor], targets: list[torch.tensor]):
        self.data: list[torch.tensor] = data
        self.targets: list[torch.tensor] = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        return self.data[idx], self.targets[idx]

def custom_collate_fn(batch):
    # Extract the input data and targets from the batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    return inputs, targets

def target_func(x: torch.tensor) -> torch.tensor:
    return torch.sin(x * x.shape[1]) + x * x.shape[0]

def noramalize(x: torch.tensor, min, max) -> torch.tensor:
    return x.sub_(min).div_(max - min) * 0.95

def denormalize(x: torch.tensor, min, max) -> torch.tensor:
    x = x / 0.95
    return x.mul_(max - min).add_(min)

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    #############################################
    #            DATA PREPROCESSING             #
    #############################################
    min_shape: int = 10
    max_shape: int = 100
    assert min_shape <= max_shape

    print(f"Generating data with shapes between {min_shape} and {max_shape}...")
    data: list[torch.tensor] = [torch.randn(1, torch.randint(min_shape, max_shape, (1,)), torch.randint(min_shape, max_shape, (1,))) for _ in range(5000)]

    print(f"Computing targets...")
    targets: list[torch.tensor] = [target_func(data[k]) for k in range(len(data))]

    # Extract the mean and standard deviation of the data and targets
    print(f"Computing mean and standard deviation of data and targets...")
    all_data = torch.cat([x.flatten() for x in data])
    max_data = all_data.max()
    min_data = all_data.min()
    del all_data
    print(f"************")
    print(f"Max data: {max_data}, min data: {min_data}")

    all_target = torch.cat([x.flatten() for x in targets])
    max_target = all_target.max()
    min_target = all_target.min()
    del all_target
    print(f"Max target: {max_target}, min target: {min_target}")
    print(f"************")

    print(f"Normalizing data and targets...")
    data_normalized = [noramalize(sample, min_data, max_data) for sample in data]
    target_normalized = [noramalize(sample, min_target, max_target) for sample in targets]

    # del data, targets

    print(f"Initializing dataset, dataloader and models...")
    # Create a dataset and dataloader
    dataset = MyDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=custom_collate_fn)

    # Initialize the models
    input_dim: int = 2  # MLP input size
    hidden_dim: int = 64  # MLP hidden layers size
    kernel_sizes: list[int] = [3, 3, 3]  # CNN kernel sizes
    num_kernels: list[int] = [16, 32, 1]  # CNN number of kernels

    mlp = KernelGeneratorMLP(input_dim, hidden_dim, kernel_sizes, num_kernels)
    conv_net = DynamicConvNet(1, num_kernels, kernel_sizes)
    mlp.to(device)
    conv_net.to(device)

    # Set up the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(mlp.parameters()) + list(conv_net.parameters()), lr=0.001)

    # Training loop
    num_epochs = 10  # Set the number of epochs

    print("Training started...")
    loss_list = []
    for epoch in range(num_epochs):
        for batch, (inputs, targets) in tqdm(enumerate(dataloader), desc="Batch progression", total=len(dataloader), unit="batch"):

            optimizer.zero_grad()  # Clear previous gradients

            batch_loss = 0

            # Forward pass through the MLP to get the kernels
            for input, target in zip(inputs, targets):
                input = input.to(device)
                target = target.to(device)

                mlp_input = torch.tensor([input.shape[1] / max_shape, input.shape[2] / max_shape], dtype=torch.float32).to(device)
                generated_kernels = mlp(mlp_input).unsqueeze(0)

                # Forward pass through the convolutional network
                output = conv_net(input, generated_kernels)

                # Compute the loss
                loss = criterion(output, target)
                batch_loss += loss
            loss_list.append(batch_loss.item())
            batch_loss.backward()

            for name, param in mlp.named_parameters():
                if param.grad is None:
                    print(f"Grad for {name}: None")

            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training complete!")
    plt.plot(loss_list)
    plt.title("Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.show()

    test_input = torch.randn(1, 1, 20, 20).to(device)
    test_input[0, 0, 10:15, 10:13] = 0.8
    test_input_normalized = noramalize(test_input, min_data, max_data)
    target_normalized = noramalize(target_func(test_input), min_data, max_data)

    mlp_input = torch.tensor([test_input.shape[2], test_input.shape[3]], dtype=torch.float32).to(device)
    generated_kernels = mlp(mlp_input).unsqueeze(0)
    output = conv_net(test_input, generated_kernels)

    
    error_map = ((output - target_normalized) * (output - target_normalized))

    min_val = min(output.min(), target_normalized.min()).item()
    max_val = max(output.max(), target_normalized.max()).item()

    plt.figure(figsize=(15, 5))

    # Plot the output
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
    plt.imshow(output[0, 0].detach().cpu().numpy(), cmap='viridis', vmin=min_val, vmax=max_val)  # Change colormap as needed
    plt.colorbar()
    plt.title('Output (Denormalized)')

    # Plot the target
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, second subplot
    plt.imshow(target_normalized[0, 0].detach().cpu().numpy(), cmap='viridis', vmin=min_val, vmax=max_val)  # Change colormap as needed
    plt.colorbar()
    plt.title('Target')

    # Plot the error map
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, third subplot
    plt.imshow(error_map[0, 0].detach().cpu().numpy(), cmap='viridis')  # Change colormap as needed
    plt.colorbar()
    plt.title('Error Map')

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()