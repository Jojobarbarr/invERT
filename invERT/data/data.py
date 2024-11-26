import logging
from torch import cat, randn, sin, randint, rand, Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from time import perf_counter

# Configure logging
logging.basicConfig(level=logging.INFO)

class IrregularDataset(Dataset):
    def __init__(self, data: list[Tensor], targets: list[Tensor]):
        self.data: list[Tensor] = data
        self.targets: list[Tensor] = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.data[idx], self.targets[idx]

def custom_collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    # Extract the input data and targets from the batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    return inputs, targets

def initialize_datasets(data: list[Tensor], target: list[Tensor], batch_size: int, test_split: float, validation_split: float) -> tuple[DataLoader, DataLoader, DataLoader]:
    logging.info("Initializing dataset, dataloader and models...") 
    dataset = IrregularDataset(data, target)

    train_size = int((1 - test_split - validation_split) * len(dataset))
    test_size = int(test_split * len(dataset))
    val_size = len(dataset) - train_size - test_size

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)

    return train_dataloader, test_dataloader, val_dataloader


def normalize(x: Tensor, min_val: float, max_val: float, lim: float=1) -> Tensor:
    return (x - min_val) / (max_val - min_val) * lim

def denormalize(x: Tensor, min_val: float, max_val: float, lim: float=1) -> Tensor:
    x = x / lim
    return x * (max_val - min_val) / lim + min_val

def target_func(x: Tensor, noise: float) -> Tensor:
    x_flip = x.flip((0, 1))
    return (sin(x * (x_flip + x.shape[0])) + x * x.shape[1]) * (1 + noise * randn(x.shape))

def generate_data(size: int, min_shape: int, max_shape: int, noise: float) -> list[tuple[Tensor, Tensor]]:
    logging.info(f"Generating data with shapes between {min_shape} and {max_shape}...")
    
    return [
        (target_func(
            target := 1000 * rand(1, randint(min_shape, max_shape, (1,)), randint(min_shape, max_shape, (1,))) + 500,
            noise
        ), target)
        for _ in range(size)
    ]

def pre_process_data(data: list[tuple[Tensor, Tensor]]) -> tuple[list[Tensor], list[Tensor], int, float, float, float, float]:
    train_data, train_targets = zip(*data)
    logging.info("Computing global min and max for normalization...")
    min_data, max_data = get_min_and_max(train_data)
    min_target, max_target = get_min_and_max(train_targets)
    max_input_shape = max(max(sample.shape[1:3]) for sample in train_data)

    logging.info("Normalizing data and targets...")
    normalize_list = lambda data, min_val, max_val: [normalize(sample, min_val, max_val) for sample in data]
    
    normalized_train_data = normalize_list(train_data, min_data, max_data)
    normalized_train_targets = normalize_list(train_targets, min_target, max_target)

    return (
        normalized_train_data, normalized_train_targets,
        max_input_shape,
        min_data, max_data,
        min_target, max_target,
    )

def get_min_and_max(data: list[Tensor]) -> tuple[float, float]:
    all_values = cat([tensor.flatten() for tensor in data])
    return all_values.min().item(), all_values.max().item()

if __name__ == "__main__":
    logging.info("Starting data generation...")
    start_time: float = perf_counter()
    data = generate_data(10000, 10, 100, 0.1)
    logging.info(f"Data generation took {perf_counter() - start_time:.4f} seconds.")
    logging.info("Starting preprocessing...")
    start_time = perf_counter()
    (
        normalized_data, normalized_target,
        max_input_shape, min_data, max_data, min_target, max_target
    ) = pre_process_data(data)
    logging.info(f"Preprocessing took {perf_counter() - start_time:.4f} seconds.")
    # Debug output
    sample_data = normalized_data[0]
    logging.info(f"Data: {sample_data}")
    logging.info(f"Data shape: {sample_data.shape}, dtype: {sample_data.dtype}")
    logging.info(f"Data mean: {sample_data.mean():.4f}, std: {sample_data.std():.4f}")

    logging.info(f"Dataset length: {len(normalized_data)}")
    logging.info(f"Data min shape: {min([sample.shape for sample in normalized_data])}, max shape: {max([sample.shape for sample in normalized_data])}")
