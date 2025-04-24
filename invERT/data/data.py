from pathlib import Path
from numpy import typing as npt
import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Optional, Tuple, TypedDict


class InvERTSample(TypedDict):
    num_electrode: torch.Tensor
    subsection_length: torch.Tensor
    array_type: torch.Tensor
    pseudosection: torch.Tensor
    norm_log_resistivity_model: torch.Tensor
    JtJ_diag: torch.Tensor

class InvERTSampleNumpy(TypedDict):
    num_electrode: npt.NDArray
    subsection_length: npt.NDArray
    array_type: npt.NDArray
    pseudosection: npt.NDArray
    norm_log_resistivity_model: npt.NDArray
    JtJ_diag: npt.NDArray

class InvERTBatch_per_cat(TypedDict):
    num_electrodes: torch.Tensor
    subsection_lengths: torch.Tensor
    array_types: torch.Tensor


class InvERTDataset(Dataset):
    def __init__(self,
                 path: Path,
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
        self.path: Path = path
        self.transform = transform
        self.samples_idx: list[str] = sorted([f.stem for f in self.path.glob('*.npz')])
        self.length: int = len(self.samples_idx)

        print(f"DATASET: Found {self.length} samples in {self.path}.")


    def __len__(self) -> int:
        """
        Return the total number of entries in the dataset.
        """
        return self.length


    def __getitem__(self,
                    index: int) -> InvERTSample:
        if index < 0:
            index = self.length + index
        if not 0 <= index < self.length:
            raise IndexError(f"Index {index} out of range for dataset of length {self.length}.")
        sample = np.load(self.path / f"{self.samples_idx[index]}.npz")
        sample = InvERTSample(
            num_electrode=torch.tensor(sample['num_electrode'], dtype=torch.float32),
            subsection_length=torch.tensor(sample['subsection_length'], dtype=torch.float32),
            array_type=torch.tensor(sample['array_type'], dtype=torch.int32),
            pseudosection=torch.tensor(sample['pseudosection'], dtype=torch.float32),
            norm_log_resistivity_model=torch.tensor(sample['norm_log_resistivity_model'], dtype=torch.float32),
            JtJ_diag=torch.tensor(sample['JtJ_diag'], dtype=torch.float32),
        )
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_per_sample(batch: list[InvERTSample]):
    return batch


def collate_pad(batch: list[InvERTSample]):
    pseudo_list = [sample['pseudosection'] for sample in batch]
    target_list = [sample['norm_log_resistivity_model'] for sample in batch]
    JtJ_diag_list = [sample['JtJ_diag'] for sample in batch]

    max_h_pseudo = max(p.shape[-2] for p in pseudo_list)
    max_w_pseudo = max(p.shape[-1] for p in pseudo_list)
    max_h_target = max(t.shape[-2] for t in target_list)
    max_w_target = max(t.shape[-1] for t in target_list)

    padded_pseudos = []
    padded_targets = []
    padded_JtJ_diag = []

    pad_value = 0

    for pseudo, target, JtJ_diag in zip(pseudo_list, target_list, JtJ_diag_list):
        # Pad pseudosection (example assumes CHW or HW format, adjust padding dims accordingly)
        h, w = pseudo.shape[-2:]
        # (padding_left, padding_right, padding_top, padding_bottom)
        padding_pseudo = (0, max_w_pseudo - w, 0, max_h_pseudo - h)
        padded_p = torch.nn.functional.pad(pseudo, padding_pseudo, mode='constant', value=pad_value)

        padded_pseudos.append(padded_p)

        # Pad target
        h, w = target.shape[-2:]
        padding_target = (0, max_w_target - w, 0, max_h_target - h)
        padded_t = torch.nn.functional.pad(target, padding_target, mode='constant', value=pad_value)

        padded_targets.append(padded_t)

        # Pad JtJ_diag
        h, w = JtJ_diag.shape[-2:]
        padding_JtJ_diag = (0, max_w_target - w, 0, max_h_target - h)
        padded_s = torch.nn.functional.pad(JtJ_diag, padding_JtJ_diag, mode='constant', value=pad_value)

        padded_JtJ_diag.append(padded_s)

    # Stack along the batch dimension
    batched_pseudos = torch.stack(padded_pseudos, dim=0)
    batched_targets = torch.stack(padded_targets, dim=0)
    batched_JtJ_diag = torch.stack(padded_JtJ_diag, dim=0)

    return {
        'pseudosection': batched_pseudos,
        'norm_log_resistivity_model': batched_targets,
        'JtJ_diag': batched_JtJ_diag,
        'array_type': torch.stack([sample['array_type'] for sample in batch]),
        'num_electrode': torch.stack([sample['num_electrode'] for sample in batch]),
        'subsection_length': torch.stack([sample['subsection_length'] for sample in batch]),
    }


def collate_pad(batch: list[InvERTSample]):
    pseudo_list = [sample['pseudosection'] for sample in batch]
    target_list = [sample['norm_log_resistivity_model'] for sample in batch]
    JtJ_diag_list = [sample['JtJ_diag'] for sample in batch]

    max_h_pseudo = max(p.shape[-2] for p in pseudo_list)
    max_w_pseudo = max(p.shape[-1] for p in pseudo_list)
    max_h_target = max(t.shape[-2] for t in target_list)
    max_w_target = max(t.shape[-1] for t in target_list)

    padded_pseudos = []
    padded_targets = []
    padded_JtJ_diag = []

    pad_value = 0

    for pseudo, target, JtJ_diag in zip(pseudo_list, target_list, JtJ_diag_list):
        # Pad pseudosection (example assumes CHW or HW format, adjust padding dims accordingly)
        h, w = pseudo.shape[-2:]
        # (padding_left, padding_right, padding_top, padding_bottom)
        padding_pseudo = (0, max_w_pseudo - w, 0, max_h_pseudo - h)
        padded_p = torch.nn.functional.pad(pseudo, padding_pseudo, mode='constant', value=pad_value)

        padded_pseudos.append(padded_p)

        # Pad target
        h, w = target.shape[-2:]
        padding_target = (0, max_w_target - w, 0, max_h_target - h)
        padded_t = torch.nn.functional.pad(target, padding_target, mode='constant', value=pad_value)

        padded_targets.append(padded_t)

        # Pad JtJ_diag
        h, w = JtJ_diag.shape[-2:]
        padding_JtJ_diag = (0, max_w_target - w, 0, max_h_target - h)
        padded_s = torch.nn.functional.pad(JtJ_diag, padding_JtJ_diag, mode='constant', value=pad_value)

        padded_JtJ_diag.append(padded_s)

    # Stack along the batch dimension
    batched_pseudos = torch.stack(padded_pseudos, dim=0)
    batched_targets = torch.stack(padded_targets, dim=0)
    batched_JtJ_diag = torch.stack(padded_JtJ_diag, dim=0)

    return {
        'pseudosection': batched_pseudos,
        'norm_log_resistivity_model': batched_targets,
        'JtJ_diag': batched_JtJ_diag,
        'array_type': torch.stack([sample['array_type'] for sample in batch]),
        'num_electrode': torch.stack([sample['num_electrode'] for sample in batch]),
        'subsection_length': torch.stack([sample['subsection_length'] for sample in batch]),
    }