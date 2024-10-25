from torch import tensor, cat, randn, sin, randint, rand, Tensor
from torch.utils.data import Dataset


class IrregularDataset(Dataset):
    def __init__(self, data: list[Tensor], targets: list[Tensor]):
        self.data: list[Tensor] = data
        self.targets: list[Tensor] = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.data[idx], self.targets[idx]
    
def custom_collate_fn(batch):
    # Extract the input data and targets from the batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    return inputs, targets

def normalize(x: Tensor, min: float, max: float, lim: float=1) -> Tensor:
    return x.sub_(min).div_(max - min) * lim

def denormalize(x: Tensor, min: float, max: float, lim: float=1) -> Tensor:
    x = x / lim
    return x.mul_(max - min).add_(min)

def target_func(x: Tensor, noise: float) -> Tensor:
    x_flip = x.flip((0, 1))
    return (sin(x * (x_flip + x.shape[0])) + x * x.shape[1]) * (1 + noise * randn(x.shape))

def generate_data(size: int, min_shape: int, max_shape: int, noise: float) -> tuple[list[Tensor], list[Tensor]]:
    print(f"Generating data with shapes between {min_shape} and {max_shape}...")
    targets: list[Tensor] = [1000 * rand(1, randint(min_shape, max_shape, (1,)), randint(min_shape, max_shape, (1,))) + 500 for _ in range(size)]

    print(f"Computing targets...")
    data: list[Tensor] = [target_func(targets[k], noise=noise) for k in range(len(targets))]

    return data, targets

def generate_validation_data(size: int, noise: float) -> tuple[list[Tensor], list[Tensor]]:
    print(f"Generating validation data with shapes between 10 and 25...")
    targets: list[Tensor] = [1000 * rand(1, randint(10, 25, (1,)), randint(10, 25, (1,))) + 500 for _ in range(size)]

    print(f"Computing targets...")
    data: list[Tensor] = [target_func(targets[k], noise=noise) for k in range(len(targets))]

    return data, targets

def pre_process_data(dataset_size: int, min_shape: int, max_shape: int, noise: float) -> tuple[float, float, float, float, list[Tensor], list[Tensor]]:
    data, targets = generate_data(dataset_size, min_shape, max_shape, noise)
    data_val, targets_val = generate_validation_data(dataset_size, noise)

    # Extract the mean and standard deviation of the data and targets
    print(f"Computing mean and standard deviation of data and targets...")
    min_data, max_data = get_min_and_max(data)
    min_target, max_target = get_min_and_max(targets)
    print(f"************")
    print(f"Max data: {max_data}, min data: {min_data}")
    print(f"Max target: {max_target}, min target: {min_target}")
    print(f"************")

    print(f"Normalizing data and targets...")
    normalized_data = [normalize(sample, min_data, max_data) for sample in data]
    normalized_target = [normalize(sample, min_target, max_target) for sample in targets]
    normalized_data_val = [normalize(sample, min_data, max_data) for sample in data_val]
    normalized_target_val = [normalize(sample, min_target, max_target) for sample in targets_val]

    return normalized_data, normalized_target, min_data, max_data, min_target, max_target, normalized_data_val, normalized_target_val

def get_min_and_max(data: list[Tensor]) -> tuple[float, float]:
    all_data = cat([x.flatten() for x in data])
    max_data = all_data.max()
    min_data = all_data.min()

    return min_data, max_data

if __name__ == "__main__":
    normalized_data, normalized_target, min_data, max_data, min_target, max_target = pre_process_data(1000, 10, 100)
    print(f"Data: {normalized_data[0]}")
    print(f"Target: {normalized_target[0]}")
    print(f"Max data: {max_data}, min data: {min_data}")
    print(f"Max target: {max_target}, min target: {min_target}")
    print(f"Data shape: {normalized_data[0].shape}")
    print(f"Target shape: {normalized_target[0].shape}")
    print(f"Data type: {normalized_data[0].dtype}")
    print(f"Target type: {normalized_target[0].dtype}")
    print(f"Data mean: {normalized_data[0].mean()}")
    print(f"Data std: {normalized_data[0].std()}")
    print(f"Target mean: {normalized_target[0].mean()}")
    print(f"Target std: {normalized_target[0].std()}")