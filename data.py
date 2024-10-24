from torch import tensor, cat, randn, sin, randint, rand
from torch.utils.data import Dataset


class IrregularDataset(Dataset):
    def __init__(self, data: list[tensor], targets: list[tensor]):
        self.data: list[tensor] = data
        self.targets: list[tensor] = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[tensor, tensor]:
        return self.data[idx], self.targets[idx]
    
def custom_collate_fn(batch):
    # Extract the input data and targets from the batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    return inputs, targets

def normalize(x: tensor, min: float, max: float, lim: float=1) -> tensor:
    return x.sub_(min).div_(max - min) * lim

def denormalize(x: tensor, min: float, max: float, lim: float=1) -> tensor:
    x = x / lim
    return x.mul_(max - min).add_(min)

def target_func(x: tensor) -> tensor:
    return sin(x * x.shape[1]) + x * x.shape[0]

def generate_data(size: int, min_shape: int, max_shape: int) -> tuple[list[tensor], list[tensor]]:
    print(f"Generating data with shapes between {min_shape} and {max_shape}...")
    data: list[tensor] = [1000 * rand(1, randint(min_shape, max_shape, (1,)), randint(min_shape, max_shape, (1,))) + 500 for _ in range(size)]

    print(f"Computing targets...")
    targets: list[tensor] = [target_func(data[k]) for k in range(len(data))]

    return data, targets

def pre_process_data(dataset_size: int, min_shape: int, max_shape: int) -> tuple[float, float, float, float, list[tensor], list[tensor]]:
    data, targets = generate_data(dataset_size, min_shape, max_shape)

    # Extract the mean and standard deviation of the data and targets
    print(f"Computing mean and standard deviation of data and targets...")
    min_data, max_data = get_mean_and_std(data)
    min_target, max_target = get_mean_and_std(targets)
    print(f"************")
    print(f"Max data: {max_data}, min data: {min_data}")
    print(f"Max target: {max_target}, min target: {min_target}")
    print(f"************")

    print(f"Normalizing data and targets...")
    normalized_data = [normalize(sample, min_data, max_data) for sample in data]
    normalized_target = [normalize(sample, min_target, max_target) for sample in targets]

    return normalized_data, normalized_target, min_data, max_data, min_target, max_target

def get_mean_and_std(data: list[tensor]) -> tuple[float, float]:
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