from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import numpy.typing as npt
from itertools import zip_longest
from collections import defaultdict
from data import invERTbatch
import warnings
import torch
import torch.nn.functional as F

def regrid(x: torch.Tensor,) -> npt.NDArray[np.float32]:
    # Get dimensions
    N, C, H, W = x.shape

    # Identify the first and last nonzero columns for each row
    row_mask = x[0, 0] != 0  # Shape: (H, W)
    col_indices = torch.arange(W, device=x.device).expand(H, W)

    col_min = torch.where(row_mask, col_indices, W).amin(dim=1)  # First nonzero column per row
    col_max = torch.where(row_mask, col_indices, -1).amax(dim=1)  # Last nonzero column per row

    # Handle empty rows (avoid invalid ranges)
    col_min[col_min == W] = 0
    col_max[col_max == -1] = W - 1

    # Compute normalized grid coordinates
    j = torch.linspace(0, 1, W, device=x.device).view(1, W)  # Column scale factor
    mapped_cols = col_min.view(H, 1) + (col_max - col_min).view(H, 1) * j  # Shape: (H, W)

    norm_x = (mapped_cols / (W - 1)) * 2 - 1  # Normalize to [-1, 1]
    norm_y = torch.linspace(-1, 1, H, device=x.device).view(H, 1).expand(H, W)  # Shape: (H, W)

    # Stack to create the grid: [N, H, W, 2]
    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0).expand(N, -1, -1, -1)

    # Perform grid sampling
    return F.grid_sample(x, grid, mode='bilinear', align_corners=True).squeeze().numpy()

def regrid_batch(batch: invERTbatch,) -> invERTbatch:
    """
    Applies the regrid function to each pseudosection in the batch.
    
    Parameters
    ----------
    batch: invERTbatch
        Batch of data containing the pseudosections.
    
    Returns
    -------
    invERTbatch
        Batch of data containing the regridded pseudosections.
    """
    num_electrodes, subsection_lengths, scheme_names, pseudosections, norm_log_resistivity_models = batch
    regridded_pseudosections = tuple((regrid(torch.from_numpy(ps).unsqueeze(0).unsqueeze(0)) for ps in pseudosections))
    return (num_electrodes, subsection_lengths, scheme_names, regridded_pseudosections, norm_log_resistivity_models)

def resize(sample: np.ndarray[np.int8],
           new_shape: tuple[int, int]
           ) -> np.ndarray[np.int8]:
    """
    Resizes a sample to a new shape using nearest neighbor interpolation.

    Parameters
    ----------
    sample : np.ndarray[np.int8]
        The sample to resize.
    new_shape : tuple[int, int]
        The new shape of the sample.

    Returns
    -------
    np.ndarray[np.int8]
        The resized sample.
    """
    src_rows, src_cols = sample.shape
    dst_rows, dst_cols = new_shape
    # Compute nearest neighbor indices
    row_indices: np.ndarray[int] = np.round(
        np.linspace(0, src_rows - 1, dst_rows)
    ).astype(int)
    col_indices: np.ndarray[int] = np.round(
        np.linspace(0, src_cols - 1, dst_cols)
    ).astype(int)

    # Use advanced indexing to select nearest neighbors
    resized_array: np.ndarray[np.int8] = sample[
        row_indices[:, None], col_indices
    ]
    return resized_array

def resize_batch(batch: invERTbatch,
                 target_size: tuple[int, int],
                 ) -> invERTbatch:
    """
    Resizes the pseudosections in the dataloader to a target size.

    Parameters
    ----------
    batch: invERTbatch
        Batch of data containing the pseudosections.
    target_size: tuple[int, int]
        Target size for the pseudosections.
    
    Returns
    -------
    invERTbatch
        Batch of data containing the resized pseudosections.
    """
    num_electrodes, subsection_lengths, scheme_names, pseudosections, norm_log_resistivity_models = batch
    resized_pseudosections = tuple((resize(ps, target_size) for ps in pseudosections))
    resized_models = tuple((resize(model, (219, 292)) for model in norm_log_resistivity_models))
    return (num_electrodes, subsection_lengths, scheme_names, resized_pseudosections, resized_models)

def log_transform(batch: invERTbatch,
                  ) -> invERTbatch:
    """
    Log transforms the pseudosections in the dataloader (log1p: log(1 + x)).
    
    Parameters
    ----------
    batch: invERTbatch
        Batch of data containing the pseudosections.
    
    Returns
    -------
    invERTbatch
        Batch of data containing the log transformed pseud
    """
    num_electrodes, subsection_lengths, scheme_names, pseudosections, norm_log_resistivity_models = batch

    # log_pseudosections = (np.where(~np.isnan(ps), np.log1p(ps), ps) for ps in pseudosections)
    log_pseudosections = []
    for ps in pseudosections:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)  # Convertit le warning en exception
                transformed_ps = np.where(~np.isnan(ps), np.log1p(ps), ps)
            log_pseudosections.append(transformed_ps)

        except RuntimeWarning as e:
            invalid_values = ps[np.isnan(np.log1p(ps)) & ~(np.isnan(ps))]  # Identifie les valeurs invalides
            print(f"Warning caught in log_transform: {e}")
            print(f"Problematic values: {invalid_values}")
            log_pseudosections.append(ps)  # Retourne la pseudo-section inchangée si erreur
    return (num_electrodes, subsection_lengths, scheme_names, tuple(log_pseudosections), norm_log_resistivity_models)


def shift(batch: invERTbatch,
          rho_app_mins: dict[str, npt.NDArray[np.float64]]
          ) -> invERTbatch:
    """
    Shifts the pseudosections in the dataloader by the minimum apparent resistivity value.
    
    Parameters
    ----------
    batch: invERTbatch
        Batch of data containing the pseudosections.
    rho_app_mins: dict[str, npt.NDArray[np.float64]]
        Dictionary containing the minimum apparent resistivity values for each array.
    
    Returns
    -------
    invERTbatch
        Batch of data containing the shifted pseudosections.
    """
    num_electrodes, subsection_lengths, scheme_names, pseudosections, norm_log_resistivity_models = batch
    shifted_pseudosections = tuple((np.where(~np.isnan(ps), ps - rho_app_mins[array][:len(ps)][:, None], ps) for ps, array in zip(pseudosections, scheme_names)))
    return (num_electrodes, subsection_lengths, scheme_names, shifted_pseudosections, norm_log_resistivity_models)


def normalize(batch: invERTbatch,
              max_value: dict[str, float]
              ) -> invERTbatch:
    """
    Normalizes the pseudosections in the dataloader by dividing by the maximum apparent resistivity value.

    Parameters
    ----------
    batch: invERTbatch
        Batch of data containing the pseudosections.
    max_value: dict[str, float]
        Dictionary containing the maximum apparent resistivity values for each array.
    
    Returns
    -------
    invERTbatch
        Batch of data containing the normalized pseudosections.
    """
    num_electrodes, subsection_lengths, scheme_names, pseudosections, norm_log_resistivity_models = batch
    normalized_num_electrodes = tuple(((num_electrode - 24) / 72 for num_electrode in num_electrodes))
    normalized_subsection_lengths = tuple(((subsection_length - 24) / 176 for subsection_length in subsection_lengths))
    normalized_array = tuple((0 if array == "wa" else 1 for array in scheme_names))
    normalized_pseudosections = tuple((np.where(~np.isnan(ps), ps / max_value[array], ps) for ps, array in zip(pseudosections, scheme_names)))

    return (normalized_num_electrodes, normalized_subsection_lengths, normalized_array, normalized_pseudosections, norm_log_resistivity_models)


def compute_sum(batch: invERTbatch,
                ) -> tuple[
                    list[npt.NDArray[np.float64]],
                    list[npt.NDArray[np.float64]],
                    list[npt.NDArray[np.int64]]
                ]:
    """
    Computes the sum and the squared sum for each row in the pseudosections.
    Counts the number of non-nan values associated with each row.
    
    Parameters
    ----------
    batch: invERTbatch
        Batch of data containing the pseudosections.
    
    Returns
    -------
    tuple[
        list[npt.NDArray[np.float64]],
        list[npt.NDArray[np.float64]],
        list[npt.NDArray[np.int64]]
    ]
        Tuple containing the sum, squared sum, and count of non-nan values for each row.
    """
    # Computes the sum for each row in the pseudosections.
    rho_app_sums: list[npt.NDArray[np.float64]] = [np.nansum(ps, axis=1) for ps in batch[3]]

    # Computes the squared sum for each row in the pseudosections.
    rho_app_sums_squared: list[npt.NDArray[np.float64]] = [np.nansum(ps ** 2, axis=1) for ps in batch[3]]

    # Counts the number of non-nan values associated with each row.
    rho_app_counts: list[npt.NDArray[np.int64]] = [np.count_nonzero(~np.isnan(ps), axis=1) for ps in batch[3]]

    return rho_app_sums, rho_app_sums_squared, rho_app_counts


def compute_min(batch: invERTbatch,
                arrays: tuple[str],
                ) -> dict[str, npt.NDArray[np.float64]]:
    """
    Computes the minimum for each row in the pseudosections.
    
    Parameters
    ----------
    batch: invERTbatch
        Batch of data containing the pseudosections.
    arrays: tuple[str]
        List containing the arrays for which the minimum apparent resistivity should be computed.
    
    Returns
    -------
    dict[str, npt.NDArray[np.float64]]
        Dictionary containing the minimum apparent resistivity values for each row.
    """
    rho_app_mins: dict[str, list[npt.NDArray[np.float64]]] = {array: [] for array in arrays}
    for ps, array in zip(batch[3], batch[2]):
        rho_app_mins[array].append(np.nanmin(ps, axis=1))
    rho_app_min: dict[str, npt.NDArray[np.float64]] = {
        array: np.array([min(mins) for mins in zip_longest(*rho_app_mins[array], fillvalue=np.inf)])
        for array in rho_app_mins
    }
    return rho_app_min


def compute_max(batch: invERTbatch,
                arrays: tuple[str],
                ) -> dict[str, npt.NDArray[np.float64]]:
    """
    Computes the maximum for each row in the pseudosections.
    
    Parameters
    ----------
    batch: invERTbatch
        Batch of data containing the pseudosections.
    arrays: tuple[str]
        List containing the arrays for which the minimum apparent resistivity should be computed.

    Returns
    -------
    dict[str, npt.NDArray[np.float64]]
        Dictionary containing the maximum apparent resistivity values for each array.
    """
    rho_app_maxs: dict[str, list[npt.NDArray[np.float64]]] = {array: [] for array in arrays}
    for ps, array in zip(batch[3], batch[2]):
        rho_app_maxs[array].append(np.nanmax(ps, axis=1))
    rho_app_max: dict[str, npt.NDArray[np.float64]] = {
        array: np.array([max(maxs) for maxs in zip_longest(*rho_app_maxs[array], fillvalue=-np.inf)])
        for array in rho_app_maxs
    }
    return rho_app_max


def compute_max_depth(batch: invERTbatch,
                      arrays: tuple[str],
                      ) -> dict[str, int]:
    """
    Computes the maximum depth of the pseudosections.
    
    Parameters
    ----------
    batch: invERTbatch
        Batch of data containing the pseudosections.
    arrays: tuple[str]
        List containing the arrays for which the minimum depth should be computed.

    Returns
    -------
    dict[str, int]
        Dictionary containing the maximum depth for each array.
    """
    max_depths: dict[str, list[int]] = {
        array: max(len(ps) for ps, array_b in zip(batch[3], batch[2]) if array_b == array)
        for array in arrays
    }
    return max_depths

def compute_min_depth(batch: invERTbatch,
                      arrays: tuple[str],
                      ) -> dict[str, int]:
    """
    Computes the minimum depth of the pseudosections.
    
    Parameters
    ----------
    batch: invERTbatch
        Batch of data containing the pseudosections.
    arrays: tuple[str]
        List containing the arrays for which the minimum depth should be computed.

    Returns
    -------
    dict[str, int]
        Dictionary containing the minimum depth for each array.
    """
    min_depths: dict[str, list[int]] = {
        array: min(len(ps) for ps, array_b in zip(batch[3], batch[2]) if array_b == array)
        for array in arrays
    }
    return min_depths


def append_by_row(batch: invERTbatch,
                  arrays: tuple[str],
            ) -> dict[str, list[npt.NDArray]]:
    """
    Appends the pseudosections by row.

    Parameters
    ----------
    batch: invERTbatch
        Batch of data containing the pseudosections.
    arrays: tuple[str]
        List containing the arrays for which the minimum depth should be computed.  
    
    Returns
    -------
    dict[str, list[npt.NDArray]]
        Dictionary containing the appended pseudosections for each row for each array.
    """
    ps_rows: list[tuple[npt.NDArray[np.float64] | None]] = list(zip_longest(*batch[3]))

    mask: list[list[npt.NDArray[np.bool_] | None]]= [
        [
            ~np.isnan(row)
            if row is not None else None
            for row in row_group
        ]
        for row_group in ps_rows
    ]
    
    concatenated_pseudosections: dict[str, list[npt.NDArray]] = {}
    for array in arrays:
        concatenated_rows: list[npt.NDArray[np.float64]] = []
        for ps_idx, row_group in enumerate(ps_rows):
            # Select rows that are not None and match the current array,
            # then apply the precomputed mask.
            selected: list[npt.NDArray[np.float64]] = [
                row[mask[ps_idx][row_idx]]
                for row_idx, row in enumerate(row_group)
                if row is not None and batch[2][row_idx] == array
            ]
            if selected:
                concatenated_rows.append(np.concatenate(selected))
        concatenated_pseudosections[array] = concatenated_rows

    return concatenated_pseudosections