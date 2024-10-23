import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # Assuming `data` and `targets` are your input and target tensors
    data: list[torch.tensor] = [torch.randn(1, torch.randint(10, 500, (1,)), torch.randint(10, 500, (1,))) for _ in range(10000)]
    targets: list[torch.tensor] = [target_func(data[k]) for k in range(len(data))]  # Example targets (100 samples, 1 channel)

    all_data = torch.cat([x.flatten() for x in data])
    mean_data = all_data.mean()
    std_data = all_data.std()
    print(f"Mean data: {mean_data}, std data: {std_data}")

    all_target = torch.cat([x.flatten() for x in targets])
    mean_target = all_target.mean()
    std_target = all_target.std()
    print(f"Mean target: {mean_target}, std target: {std_target}")

    data_normalized = [sample.sub_(mean_data).div_(std_data) for sample in data]
    target_normalized = [sample.sub_(mean_target).div_(std_target) for sample in targets]

    """
    plt.imshow(data[0][0].detach().numpy())
    plt.colorbar()
    plt.show()
    plt.imshow(targets[0][0].detach().numpy())
    plt.colorbar()
    plt.show()
    """
    # Create a dataset and dataloader
    dataset = MyDataset(data_normalized, target_normalized)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=custom_collate_fn)

    # Initialize the models
    input_dim: int = 2  # MLP input size
    hidden_dim: int = 64
    kernel_sizes: list[int] = [3, 3, 3]
    num_kernels: list[int] = [16, 32, 1]

    mlp = KernelGeneratorMLP(input_dim, hidden_dim, kernel_sizes, num_kernels)
    conv_net = DynamicConvNet(1, num_kernels, kernel_sizes)
    mlp.to(device)
    conv_net.to(device)

    # Set up the loss function and optimizer
    criterion = nn.MSELoss()  # Choose an appropriate loss function
    optimizer = optim.Adam(list(mlp.parameters()) + list(conv_net.parameters()), lr=0.01)

    # Training loop
    num_epochs = 10  # Set the number of epochs

    print("Training started...")
    for epoch in range(num_epochs):
        for batch, (inputs, targets) in tqdm(enumerate(dataloader), desc="Batch progression", total=len(dataloader), unit="batch"):
            optimizer.zero_grad()  # Clear previous gradients

            batch_loss = 0

            # Forward pass through the MLP to get the kernels
            for input, target in zip(inputs, targets):
                input = input.to(device)
                target = target.to(device)

                mlp_input = torch.tensor([input.shape[1], input.shape[2]], dtype=torch.float32).to(device)
                generated_kernels = mlp(mlp_input).unsqueeze(0)

                # Forward pass through the convolutional network
                output = conv_net(input, generated_kernels)

                # Compute the loss
                loss = criterion(output, target)
                batch_loss += loss

            batch_loss.backward()

            for name, param in mlp.named_parameters():
                if param.grad is None:
                    print(f"Grad for {name}: None")

            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training complete!")

    test_input = torch.randn(1, 1, 20, 20).to(device)
    test_input[0, 0, 10:15, 10:12] = 0.8
    test_input_normalized = test_input.sub_(mean_data).div_(std_data)
    target = target_func(test_input)

    mlp_input = torch.tensor([test_input.shape[2], test_input.shape[3]], dtype=torch.float32).to(device)
    generated_kernels = mlp(mlp_input).unsqueeze(0)
    output = conv_net(test_input, generated_kernels)
    output_denormalized = output.mul_(std_target).add_(mean_target)

    
    error_map = ((output_denormalized - target) * (output_denormalized - target))

    min_val = min(output_denormalized.min(), target.min()).item()
    max_val = max(output_denormalized.max(), target.max()).item()

    plt.figure(figsize=(15, 5))

    # Plot the output
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
    plt.imshow(output_denormalized[0, 0, 1:-1, 1:-1].detach().cpu().numpy(), cmap='viridis', vmin=min_val, vmax=max_val)  # Change colormap as needed
    plt.colorbar()
    plt.title('Output (Denormalized)')

    # Plot the target
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, second subplot
    plt.imshow(target[0, 0, 1:-1, 1:-1].detach().cpu().numpy(), cmap='viridis', vmin=min_val, vmax=max_val)  # Change colormap as needed
    plt.colorbar()
    plt.title('Target')

    # Plot the error map
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, third subplot
    plt.imshow(error_map[0, 0, 1:-1, 1:-1].detach().cpu().numpy(), cmap='viridis')  # Change colormap as needed
    plt.colorbar()
    plt.title('Error Map')

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()