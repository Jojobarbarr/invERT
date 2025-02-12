import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio
from torchvision import transforms

def transform_and_visualize(x, noise=0.1):
    x_flip = torch.flip(x, dims=(2, 3))
    x_width = x.shape[2]
    x_height = x.shape[3]
    
    step1 = x_flip + x
    # Create a grid of (x,y) coordinates for each pixel.
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, x_width),
            torch.linspace(0, 1, x_height)
        )
    )

    step2 = (step1 + (x_width / 4) * torch.sin((2 * torch.pi * grid[1]) + torch.sin(2 * torch.pi * grid[0] / x_width))) % x_width 
    + ((x_height / 4) * torch.cos((2 * torch.pi * grid[0]) + torch.cos(2 * torch.pi * grid[1] / x_height))) % x_height
    
    
    noise_tensor = (1 + noise * torch.randn_like(x))
    final_output = step2 * noise_tensor
    
    return x, x_flip, step1, step2, final_output
# Load an interpretable image (e.g., a cat face or a tree) as a tensor
image_path = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/256px-Cat03.jpg'  # Example cat image
image = imageio.v3.imread(image_path, mode='F')
image = torch.tensor(image, dtype=torch.float32)
image = image.unsqueeze(0).squeeze(0)  # Ensure correct shape
image = (image - image.min()) / (image.max() - image.min()) * 1000 + 500  # Normalize to 500-1500 range
x = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

# Apply transformations
original, flipped, step1, step2, final = transform_and_visualize(x)

# Function to plot images
def plot_tensor_images(tensors, titles, cmap='gray'):
    fig, axes = plt.subplots(1, len(tensors), figsize=(25, 5))
    for ax, tensor, title in zip(axes, tensors, titles):
        ax.imshow(tensor[0, 0].detach().cpu().numpy(), cmap=cmap, aspect='auto')
        ax.set_title(title)
        ax.axis('off')
    plt.show()

# Plot all transformation steps
plot_tensor_images(
    [original, flipped, step1, step2, final],
    ['Original', 'Flipped', 'Step 1', 'Step 2', 'Final with Noise']
)
