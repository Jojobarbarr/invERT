from pathlib import Path
import io

import lmdb
import torch
from torch.utils.data import Dataset, get_worker_info, random_split, Subset

from typing import Optional, Tuple, TypedDict


class InvERTSample(TypedDict):
    num_electrode: torch.Tensor
    subsection_length: torch.Tensor
    array_type: torch.Tensor
    pseudosection: torch.Tensor
    norm_log_resistivity_model: torch.Tensor

class InvERTBatch_per_cat(TypedDict):
    num_electrodes: torch.Tensor
    subsection_lengths: torch.Tensor
    array_types: torch.Tensor


def worker_init_fn(worker_id):
    """Initializes the LMDB environment for each worker process."""
    worker_info = get_worker_info()
    dataset_obj = worker_info.dataset  # The dataset copy in this worker process

    # Access the underlying dataset (your LMDBDataset instance)
    # It's good practice to check if it's actually a Subset first
    if isinstance(dataset_obj, Subset):
        original_dataset = dataset_obj.dataset 
    else:
        # If the DataLoader was given the original dataset directly
        original_dataset = dataset_obj 
    # Create and store the LMDB environment directly in the worker's dataset copy
    original_dataset._env = lmdb.open(
        original_dataset.lmdb_path,
        readonly=True,
        lock=False,  # OK for read-only
        readahead=original_dataset.readahead,
        meminit=False, # Good practice
    )


class LMDBDataset(Dataset):
    def __init__(self,
                 lmdb_path: Path | str,
                 readahead: bool = False,
                 transform: Optional[callable] = None,
                 ):
        """
        Initialize the LMDB dataset.
        
        Parameters
        ----------
        lmdb_path : Path or str
            Path to the LMDB directory.
        readahead : bool, optional
            Whether to use readahead for LMDB, defaults to 'False' which will disable the OS filesystem readahead mechanism, which may improve random read performance.
            Use 'True' for access patterns that are more sequential.
        transform : callable, optional
            A function/transform to apply to the data.
        """
        super().__init__()
        self.lmdb_path: str = str(lmdb_path) if isinstance(lmdb_path, Path) else lmdb_path
        self._env: Optional[lmdb.Environment] = None
        self.readahead: bool = readahead
        self.transform = transform

        # Get length once
        env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,  # Ok since readonly
            readahead=self.readahead,
            meminit=False,  # We don't have sensitive data
        )
        with env.begin() as txn:
            self.length = txn.stat()["entries"]
        env.close()

        print(f"DATASET: Found {self.length} samples in {self.lmdb_path}.")


    def __len__(self) -> int:
        """
        Return the total number of entries in the dataset.
        """
        return self.length


    def __getitem__(self,
                    index: int) -> InvERTSample:
        if self._env is None:
            # This should ideally not happen if used with DataLoader and worker_init_fn
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=self.readahead,
                meminit=False,
            )
            print(f"Warning: LMDB environment initialized lazily in getitem. Worker ID: {get_worker_info().id if get_worker_info() else 'Main'}")


        if index < 0:
            index = self.length + index
        if not 0 <= index < self.length:
            raise IndexError(f"Index {index} out of range for dataset of length {self.length}.")

        key = f"{index:08d}".encode('ascii')
        try:
            with self._env.begin() as txn:
                data_bytes: Optional[bytes] = txn.get(key)

            if data_bytes is None:
                 raise KeyError(f"Key {key.decode()} not found in LMDB {self.lmdb_path}") # Or handle differently

            buffer = io.BytesIO(data_bytes)
            # Wrap torch.load in try-except to catch deserialization errors
            try:
                sample: InvERTSample = torch.load(buffer, map_location='cpu') # map_location might be important
            except Exception as e_load:
                raise RuntimeError(f"Failed to torch.load data for key {key.decode()} (index {index})") from e_load

            return self.transform(sample) if self.transform else sample
        except lmdb.Error as e_lmdb:
             # Catch LMDB specific errors during transaction/get
             raise RuntimeError(f"LMDB error accessing key {key.decode()} (index {index})") from e_lmdb


    def close(self):
        """
        Close the LMDB environment if it is open.
        """
        if self._env is not None:
            self._env.close()
            self._env = None


    def __del__(self):
        """
        Destructor to ensure the LMDB environment is closed.
        """
        self.close()


def lmdb_collate_fn(batch: list[InvERTSample]) -> list[InvERTSample]:
    return batch


def lmdb_collate_fn_per_cat(batch: list[InvERTSample]) -> InvERTBatch_per_cat:
    num_electrodes = torch.stack([item['num_electrode'] for item in batch])
    subsection_length = torch.stack([item['subsection_length'] for item in batch])
    array_type = torch.stack([item['array_type'] for item in batch])

    batch = InvERTBatch_per_cat(
        num_electrodes=num_electrodes,
        subsection_lengths=subsection_length,
        array_types=array_type,
    )

    return batch
    
def collate_pad(batch: list[InvERTSample]):
    pseudo_list = [sample['pseudosection'] for sample in batch]
    target_list = [sample['norm_log_resistivity_model'] for sample in batch]

    max_h_pseudo = max(p.shape[-2] for p in pseudo_list)
    max_w_pseudo = max(p.shape[-1] for p in pseudo_list)
    max_h_target = max(t.shape[-2] for t in target_list)
    max_w_target = max(t.shape[-1] for t in target_list)

    padded_pseudos = []
    pseudo_masks = []
    padded_targets = []
    target_masks = []

    pad_value = 0

    for pseudo, target in zip(pseudo_list, target_list):
        # Pad pseudosection (example assumes CHW or HW format, adjust padding dims accordingly)
        h, w = pseudo.shape[-2:]
        # (padding_left, padding_right, padding_top, padding_bottom)
        padding_pseudo = (0, max_w_pseudo - w, 0, max_h_pseudo - h)
        padded_p = torch.nn.functional.pad(pseudo, padding_pseudo, mode='constant', value=pad_value)
        mask_p = torch.ones_like(pseudo, dtype=torch.bool) # Mask for original data
        mask_p = torch.nn.functional.pad(mask_p, padding_pseudo, mode='constant', value=False) # Pad mask with False

        padded_pseudos.append(padded_p)
        pseudo_masks.append(mask_p)

        # Pad target
        h, w = target.shape[-2:]
        padding_target = (0, max_w_target - w, 0, max_h_target - h)
        padded_t = torch.nn.functional.pad(target, padding_target, mode='constant', value=pad_value)
        mask_t = torch.ones_like(target, dtype=torch.bool) # Mask for original data
        mask_t = torch.nn.functional.pad(mask_t, padding_target, mode='constant', value=False) # Pad mask with False

        padded_targets.append(padded_t)
        target_masks.append(mask_t)

    # Stack along the batch dimension
    batched_pseudos = torch.stack(padded_pseudos, dim=0)
    batched_pseudo_masks = torch.stack(pseudo_masks, dim=0)
    batched_targets = torch.stack(padded_targets, dim=0)
    batched_target_masks = torch.stack(target_masks, dim=0)

    return {
        'pseudosection': batched_pseudos,
        'pseudosection_mask': batched_pseudo_masks,
        'norm_log_resistivity_model': batched_targets,
        'target_mask': batched_target_masks,
        'array_type': torch.stack([sample['array_type'] for sample in batch]),
        'num_electrode': torch.stack([sample['num_electrode'] for sample in batch]),
    }