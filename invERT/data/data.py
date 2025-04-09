import pickle
from pathlib import Path
import io

import lmdb
import numpy as np
import numpy.typing as npt
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

    
    def _get_env(self) -> lmdb.Environment:
        """
        Get or create the LMDB environment for the current worker.

        Returns
        -------
        lmdb.Environment
            The LMDB environment.
        """
        if self._env is None:
            worker_info = get_worker_info()
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,  # Ok since readonly
                readahead=self.readahead,
                meminit=False,
                max_readers=128 if worker_info else 1,
            )
        return self._env


    def __len__(self) -> int:
        """
        Return the total number of entries in the dataset.
        """
        return self.length


    def __getitem__(self,
                    index: int) -> InvERTSample:
        """
        Retrieve an item from the dataset.

        Parameters
        ----------
        index : int
            The index of the item to retrieve.
        Returns
        -------
        invERTbatch
            The item retrieved from the dataset.
        """
        if index < 0:
            index = self.length + index
        if not 0 <= index < self.length:
            raise IndexError(f"Index {index} out of range for dataset of length {self.length}.")
 
        key = f"{index:08d}".encode('ascii')
        with self._get_env().begin() as txn:
            data: bytes = txn.get(key)
        buffer = io.BytesIO(data)
        sample: InvERTSample = torch.load(buffer)
        return self.transform(sample) if self.transform else sample


    def split(self,
              test_split: float,
              val_split: float) -> Tuple[Subset['LMDBDataset'], Subset['LMDBDataset'], Subset['LMDBDataset']]:
        """
        Split the dataset into training, validation, and test sets.
        Parameters
        ----------
        test_split : float
            Proportion of the dataset to include in the test split.
        val_split : float
            Proportion of the dataset to include in the validation split.
        
        Returns
        -------
        Tuple[Subset[LMDBDataset], Subset[LMDBDataset], Subset[LMDBDataset]]
            The training, validation, and test subsets.

        """
        total_length = self.length
        test_size = int(test_split * total_length)
        val_size = int(val_split * total_length)
        train_size = total_length - test_size - val_size

        return random_split(
            self,
            [
                train_size,
                test_size,
                val_size,
            ]
        )


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