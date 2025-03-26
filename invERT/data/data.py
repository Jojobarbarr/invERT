import logging
import lmdb
import pickle
from torch import cat, randint, rand, Tensor, flip, sin, cos
import torch
# from torch.functional import F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
# Configure logging
logging.basicConfig(level=logging.INFO)


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path: Path):
        # Open the LMDB environment in read-only mode.
        self.env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        # Retrieve the total number of entries.
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
            print(f"Found {self.length} samples in the LMDB dataset.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Build the key as used during saving (zero-padded).
        key = f"{index:08d}".encode('ascii')
        with self.env.begin() as txn:
            data = txn.get(key)
        return pickle.loads(data)

    def split(self,
              test_split: float,
              val_split: float = 0):
        return random_split(self,
                            [int((1 - test_split - val_split) * len(self)),
                             int(test_split * len(self)),
                                int(val_split * len(self))])


def lmdb_custom_collate_fn(batch):
    """
    Custom collate function to handle batches with heterogeneous items.

    Each item in batch is a tuple: (int, int, str, np.ndarray, np.ndarray).
    For fields with varying shapes (the numpy arrays), we keep them as lists.
    For the other items, you can choose to stack or keep as is.
    """
    num_electrodes, subsection_lengths, scheme_names, \
        pseudosections, norm_log_resistivity_models = zip(*batch)

    num_electrodes = torch.tensor(num_electrodes, dtype=torch.int64)
    subsection_lengths = torch.tensor(subsection_lengths, dtype=torch.int64)

    return (
        num_electrodes,
        subsection_lengths,
        scheme_names,
        pseudosections,
        norm_log_resistivity_models
    )


class CustomDataset(Dataset):
    def __init__(self,
                 data: Tensor,
                 ):
        """
        Initialize the dataset with data that is either data or target.

        The dataset is composed of data. The data is a tensor of shape
        (num_samples, 1, width, height).

        @param data: The data tensor.
        """
        self.data: Tensor = data

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        @return: The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self,
                    idx: int
                    ) -> Tensor:
        assert idx < len(self), \
            (f"Index {idx} is out of bounds for the dataset with length "
             f"{len(self)}.")
        return self.data[idx]


class IrregularDataset(Dataset):
    def __init__(self,
                 data: list[Tensor],
                 targets: list[Tensor]):
        """
        Initialize the dataset with data and targets.

        The dataset is composed of data and targets. The data and targets are
        lists of tensors. The length of the lists is the number of sub-groups
        in the dataset. The data and targets are of shape
        (num_samples_per_sub_group, 1, width, height) with varying width and
        height for each sub-group.

        @param data: The list of data tensors.
        @param targets: The list of target tensors.
        """
        self.data: list[CustomDataset] = [
            CustomDataset(data_chunk) for data_chunk in data
        ]
        self.targets: list[CustomDataset] = [
            CustomDataset(target_chunk) for target_chunk in targets]

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        The total number of samples is the product of the number of sub-groups
        and the number of samples per sub-group.

        @return: The total number of samples in the dataset.
        """
        return sum(len(sub_group) for sub_group in self.data)

    def len_sub_group(self) -> int:
        """
        Return the number of samples per sub-group.

        @return: The number of samples per sub-group.
        """
        return len(self.data[0])

    def __getitem__(self,
                    idx: int,
                    ) -> tuple[Tensor, Tensor]:
        cumulative_length = 0
        if isinstance(idx, int):  # Handle single index
            for group_data, group_targets in zip(self.data, self.targets):
                if idx < cumulative_length + len(group_data):
                    local_idx = idx - cumulative_length
                    return group_data[local_idx], group_targets[local_idx]
                cumulative_length += len(group_data)
            raise IndexError(f"Index {idx} out of bounds")
        elif isinstance(idx, slice):  # Handle slice
            indices = range(*idx.indices(len(self)))
            samples = [self[i] for i in indices]
            return samples
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")


def initialize_datasets(data: list[Tensor],
                        val_data: list[Tensor],
                        target: list[Tensor],
                        val_target: list[Tensor],
                        batch_size: int,
                        batch_mixture: int,
                        num_sub_group: int,
                        sub_group_size: int,
                        test_split: float,
                        ) -> tuple[
                            tuple[DataLoader],
                            tuple[DataLoader],
                            tuple[DataLoader]
]:
    logging.info("Initializing dataset, dataloader and models...")
    dataset = IrregularDataset(data, target)
    val_dataset = IrregularDataset(val_data, val_target)

    mini_batch_size: int = batch_size // batch_mixture
    sub_groups_size: int = dataset.len_sub_group()
    val_sub_groups_size: int = val_dataset.len_sub_group()
    train_size: int = sub_group_size
    test_size: int = int(test_split * train_size)

    train_dataloaders: list[DataLoader] = []
    test_dataloaders: list[DataLoader] = []
    val_dataloaders: list[DataLoader] = []

    for group_idx in range(num_sub_group):
        start_idx = group_idx * sub_groups_size
        val_start_idx = group_idx * val_sub_groups_size
        end_idx = (group_idx + 1) * sub_groups_size
        val_end_idx = (group_idx + 1) * val_sub_groups_size
        train_dataset, test_dataset = random_split(
            dataset[start_idx:end_idx], [train_size, test_size])

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=mini_batch_size,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=mini_batch_size,
            shuffle=True,)
        val_dataloader = DataLoader(
            val_dataset[val_start_idx:val_end_idx],
            batch_size=mini_batch_size,
            shuffle=True,)

        train_dataloaders.append(train_dataloader)
        test_dataloaders.append(test_dataloader)
        val_dataloaders.append(val_dataloader)

    return train_dataloaders, test_dataloaders, val_dataloaders


def normalize(x: Tensor,
              min_val: float,
              max_val: float,
              scaling: float = 0.95
              ) -> Tensor:
    """
    Normalize a list of tensors.

    The normalization is done by subtracting the minimum value and dividing by
    the maximum value. The scaling parameter can be used to scale the values
    to a different range, for example between 0 and 0.95.

    @param x: The list of tensors to normalize.
    @param min_val: The minimum value of the tensors.
    @param max_val: The maximum value of the tensors.
    @param scaling: The scaling factor.
    @return: The normalized list of tensors.
    """
    return (x - min_val) / (max_val - min_val) * scaling


def denormalize(
        x: Tensor,
        min_val: float,
        max_val: float,
        lim: float = 1) -> Tensor:
    x = x / lim
    return x * (max_val - min_val) / lim + min_val


def target_func(x: Tensor,
                noise: float
                ) -> Tensor:
    # 1. Flip the image.
    x = flip(x, dims=(2, 3))

    # 2. Non injective transformation, metadata dependant.
    x_width = x.shape[2]
    x_height = x.shape[3]
    x = sin(x ** 2) % x_width + cos(x * 3) % x_height
    return x


def generate_data(num_samples: int,
                  num_sub_groups: int,
                  data_min_size: int,
                  data_max_size: int,
                  noise: float,
                  num_val_samples: int,
                  ) -> tuple[list[tuple[Tensor, Tensor]],
                             list[tuple[Tensor, Tensor]]]:
    """
    Generate a dataset composed of sub-groups of different data shapes.

    Each sample in a sub-group has the same shape. The dataset is composed of
    sub-groups of different data shapes.

    @param num_samples: Total number of samples in the dataset.
    @param num_sub_groups: Number of sub-groups in the dataset.
    @param data_min_size: Minimum size of the data.
    @param data_max_size: Maximum size of the data.
    @param noise: Noise level to apply to the target function.
    @param num_val_samples: Number of samples to use for validation.
    @return: The generated dataset, a list of tuples (data, target) and a list
    of tuples (val_data, val_target). The list has a length of num_sub_groups.
    """
    logging.info(
        f"Generating data with shapes between "
        f"{data_min_size} and {data_max_size}...")

    data: list[tuple[Tensor, Tensor]] = []
    val_data: list[tuple[Tensor, Tensor]] = []

    # The dataset is composed of sub-groups of different data shapes, but each
    # sample in a sub-group has the same shape.
    sub_group_sample_size: int = num_samples // num_sub_groups
    for _ in range(num_sub_groups):
        # Generate a sub-dataset with a uniform random shape
        width: int = randint(data_min_size, data_max_size, (1,)).item()
        width_val: int = randint(data_min_size, data_max_size, (1,)).item()
        height: int = randint(data_min_size, data_max_size, (1,)).item()
        height_val: int = randint(data_min_size, data_max_size, (1,)).item()
        data_shape: tuple[int, int, int, int] = (sub_group_sample_size,
                                                 1,  # Number of channels
                                                 width,
                                                 height)
        val_data_shape: tuple[int, int, int, int] = (num_val_samples,
                                                     1,  # Number of channels
                                                     width_val,
                                                     height_val)

        # Generate a tensor with shape data_shape, with random values between
        # 500 and 1500
        target_chunk: Tensor = 1000 * rand(data_shape) + 500
        val_target_chunk: Tensor = 1000 * rand(val_data_shape) + 500

        # Generate the target tensor by applying the target function to the
        # data tensor
        data_chunk: Tensor = target_func(target_chunk, noise)
        val_data_chunk: Tensor = target_func(val_target_chunk, noise)

        data.append((data_chunk, target_chunk))
        val_data.append((val_data_chunk, val_target_chunk))

    return data, val_data


def pre_process_data_lmdb(dataloader: DataLoader):
    pass


def pre_process_data(data: list[tuple[Tensor, Tensor]],
                     ) -> tuple[list[Tensor],
                                list[Tensor],
                                float,
                                float,
                                float,
                                float]:
    """
    Pre-process the data and targets.

    The data and targets are normalized using the global minimum and maximum
    values. The function returns the normalized data and targets, as well as
    the minimum and maximum values used for normalization to use them for
    denormalization.

    @param data: The list of tuples (data, target). The length of the list is
    the number of sub-groups in the dataset.
    @return: The normalized data and targets, the minimum and maximum values
    used for normalization.
    """
    # Unzip the data and combines the data and target tensors in two lists
    train_data, train_targets = zip(*data)
    # train_data and train_targets are lists of tensors of shape
    # (num_samples_per_sub_group, 1, width, height) with varying width and
    # height for each sub-group.

    logging.info("Computing global min and max for normalization...")

    # Compute the global min and max values for normalization.
    min_data, max_data = get_min_and_max(train_data)
    min_target, max_target = get_min_and_max(train_targets)

    logging.info("Normalizing data and targets...")

    # Normalize the data and targets.
    normalized_train_data: list[Tensor] = [
        normalize(subgroup, min_data, max_data) for subgroup in train_data
    ]
    normalized_train_targets: list[Tensor] = [
        normalize(subgroup, min_target, max_target)
        for subgroup in train_targets
    ]

    return (
        normalized_train_data,
        normalized_train_targets,
        min_data,
        max_data,
        min_target,
        max_target,
    )


def get_min_and_max(data: list[Tensor]
                    ) -> tuple[float, float]:
    """
    Get the minimum and maximum values of a list of tensors.

    The tensors are flattened and concatenated. The minimum
    and maximum values are computed on the resulting tensor.

    @param data: The list of tensors.
    @return: The minimum and maximum values.
    """
    all_values: Tensor = cat([tensor.flatten() for tensor in data])
    return all_values.min().item(), all_values.max().item()
