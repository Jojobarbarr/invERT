import logging
from torch import cat, randn, sin, rand, Tensor
from torch.utils.data import Dataset, DataLoader, random_split

# Configure logging
logging.basicConfig(level=logging.INFO)


class IrregularDataset(Dataset):
    def __init__(self, data: list[Tensor], targets: list[Tensor]):
        self.data: list[Tensor] = data
        self.targets: list[Tensor] = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
                self,
                idx: int | slice
            ) -> tuple[Tensor, Tensor] | list[tuple[Tensor, Tensor]]:
        if isinstance(idx, slice):
            data_slice = self.data[idx]
            targets_slice = self.targets[idx]
            # Return as a list of tuples
            return list(zip(data_slice, targets_slice))
        return self.data[idx], self.targets[idx]


def custom_collate_fn(
        batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    # Extract the input data and targets from the batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    return inputs, targets


def initialize_datasets(
            data: list[Tensor],
            target: list[Tensor],
            batch_size: int,
            batch_mixture: int,
            test_split: float,
            validation_split: float,
            num_sub_group: int
        ) -> tuple[
            tuple[DataLoader],
            tuple[DataLoader],
            tuple[DataLoader]
        ]:
    logging.info("Initializing dataset, dataloader and models...")
    dataset = IrregularDataset(data, target)

    mini_batch_size: int = batch_size // batch_mixture
    sub_groups_size: int = len(dataset) // num_sub_group
    train_size: int = int((1 - test_split - validation_split)
                          * sub_groups_size)
    test_size: int = int(test_split * sub_groups_size)
    val_size: int = sub_groups_size - train_size - test_size

    train_dataloaders: list[DataLoader] = []
    test_dataloaders: list[DataLoader] = []
    val_dataloaders: list[DataLoader] = []

    for group_idx in range(num_sub_group):
        start_idx: int = group_idx * sub_groups_size
        end_idx: int = start_idx + sub_groups_size

        train_dataset, test_dataset, val_dataset = random_split(
            dataset[start_idx:end_idx], [train_size, test_size, val_size])

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=mini_batch_size,
            shuffle=True,)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=mini_batch_size,
            shuffle=True,)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=mini_batch_size,
            shuffle=True,)

        train_dataloaders.append(train_dataloader)
        test_dataloaders.append(test_dataloader)
        val_dataloaders.append(val_dataloader)

    return train_dataloaders, test_dataloaders, val_dataloaders


def normalize(
        x: Tensor,
        min_val: float,
        max_val: float,
        lim: float = 1) -> Tensor:
    return (x - min_val) / (max_val - min_val) * lim


def denormalize(
        x: Tensor,
        min_val: float,
        max_val: float,
        lim: float = 1) -> Tensor:
    x = x / lim
    return x * (max_val - min_val) / lim + min_val


def target_func(x: Tensor, noise: float) -> Tensor:
    x_flip = x.flip((0, 1))
    return (sin(x * (x_flip + x.shape[0])) + x
            * x.shape[1]) * (1 + noise * randn(x.shape))


def generate_data(
            size: int,
            sub_groups: int,
            min_shape: int,
            max_shape: int,
            noise: float
        ) -> list[tuple[Tensor, Tensor]]:
    logging.info(
        f"Generating data with shapes between {min_shape} and {max_shape}...")
    data: list[tuple[Tensor, Tensor]] = []
    for _ in range(sub_groups):
        data_shape: tuple[int, int] = (size // sub_groups,
                                       1,
                                       min_shape,
                                       max_shape)
        data_chunk: Tensor = 1000 * rand(data_shape) + 500
        target_chunk: Tensor = target_func(
            data_chunk,
            noise
        )
        data.append((data_chunk, target_chunk))
    return data


def pre_process_data(
        data: list[tuple[Tensor, Tensor]]
    ) -> tuple[list[Tensor],
               list[Tensor],
               int,
               float,
               float,
               float,
               float]:
    train_data, train_targets = zip(*data)
    logging.info("Computing global min and max for normalization...")
    min_data, max_data = get_min_and_max(train_data)
    min_target, max_target = get_min_and_max(train_targets)
    max_input_shape = max(max(sample.shape[1:3]) for sample in train_data)

    logging.info("Normalizing data and targets...")
    normalized_train_data: list[Tensor] = [
        normalize(sample, min_data, max_data) for subgroup in train_data
        for sample in subgroup
    ]
    normalized_train_targets: list[Tensor] = [
        normalize(sample, min_target, max_target)
        for subgroup in train_targets for sample in subgroup
    ]

    return (
        normalized_train_data, normalized_train_targets,
        max_input_shape,
        min_data, max_data,
        min_target, max_target,
    )


def get_min_and_max(
            data: list[Tensor]
        ) -> tuple[float, float]:
    all_values = cat([tensor.flatten() for tensor in data])
    return all_values.min().item(), all_values.max().item()
