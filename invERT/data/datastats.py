import numpy as np
import numpy.typing as npt
from argparse import ArgumentParser, Namespace
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from data import LMDBDataset, lmdb_custom_collate_fn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from collections import defaultdict, Counter
mp.set_sharing_strategy('file_system')


def process_batch(batch: tuple[torch.Tensor,
                               torch.Tensor,
                               list[str],
                               list[npt.NDArray[np.float64]],
                               list[npt.NDArray[np.float64]]
                               ]
                  ) -> tuple[
                      npt.NDArray[np.bool_],
                      npt.NDArray[np.int32],
                      dict[str, npt.NDArray[np.float64]],
                      dict[str, npt.NDArray[np.uint64]]
]:
    """
    Process a single batch: determine array types and depth levels.

    Parameters
    ----------
    batch: tuple[
        torch.Tensor,
        torch.Tensor,
        list[str],
        list[npt.NDArray[np.float64]],
        list[npt.NDArray[np.float64]]
    ]
        A batch of data from the DataLoader. The batch contains:
          - tensor: Tensor of number of electrodes
          - tensor: Tensor of model length
          - list[str]: List of pseudo section array types.
          - list[npt.NDArray[np.float64]]: List of pseudosections.
          - list[npt.NDArray[np.float64]]: List of resistivity models.

    Returns
    -------
    tuple[
        npt.NDArray[np.bool_],
        npt.NDArray[np.uint32],
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.uint64]]
    ]
        A tuple containing:
          - array_types: Boolean numpy array where each element indicates if
          corresponding entry equals "wa".
          - depth_levels: Numpy array of depth level values.
          - rho_app_means: Dictionary of mean apparent resistivity values for
          each array type.
          - num_pixels: Dictionary of number of pixels for each array type.
    """
    array_types: npt.NDArray[np.bool_] = np.array(
        [array == "wa" for array in batch[2]], dtype=np.bool_
    )
    depth_levels: npt.NDArray[np.uint32] = np.array(
        [pseudosection.shape[0] for pseudosection in batch[3]]
    )

    max_depths = {
        "wa": depth_levels[array_types].max(initial=0),
        "slm": depth_levels[~array_types].max(initial=0)
    }

    rho_app_means: dict[str, npt.NDArray[np.float64]] = {
        array: np.zeros((max_depth,), dtype=np.float64)
        for array, max_depth in max_depths.items()
    }
    rho_app_sq_sums: dict[str, npt.NDArray[np.float64]] = {
        array: np.zeros((max_depth,), dtype=np.float64)
        for array, max_depth in max_depths.items()
    }
    rho_app_maxs: dict[str, npt.NDArray[np.float64]] = {
        array: np.zeros((max_depth,), dtype=np.float64)
        for array, max_depth in max_depths.items()
    }
    rho_app_mins: dict[str, npt.NDArray[np.float64]] = {
        array: np.zeros((max_depth,), dtype=np.float64)
        for array, max_depth in max_depths.items()
    }
    num_pixels: dict[str, npt.NDArray[np.uint64]] = {
        array: np.zeros((max_depth,), dtype=np.uint64)
        for array, max_depth in max_depths.items()
    }

    row_sums = defaultdict(
        lambda: np.zeros(max(max_depths.values()), dtype=np.float64)
    )
    row_sums_sq = defaultdict(
        lambda: np.zeros(max(max_depths.values()), dtype=np.float64)
    )
    row_counts = defaultdict(
        lambda: np.zeros(max(max_depths.values()), dtype=np.uint64)
    )

    for array_type, pseudosection in zip(batch[2], batch[3]):
        valid_mask = ~np.isnan(pseudosection)
        row_sums[array_type][: pseudosection.shape[0]] += np.nansum(
            pseudosection, axis=1
        )
        row_sums_sq[array_type][: pseudosection.shape[0]] += np.nansum(
            pseudosection ** 2, axis=1
        )
        rho_app_maxs[array_type][: pseudosection.shape[0]] = np.maximum(
            rho_app_maxs[array_type][: pseudosection.shape[0]],
            np.nanmax(pseudosection, axis=1),
        )
        rho_app_mins[array_type][: pseudosection.shape[0]] = np.minimum(
            rho_app_mins[array_type][: pseudosection.shape[0]],
            np.nanmin(pseudosection, axis=1),
        )
        row_counts[array_type][: pseudosection.shape[0]] += valid_mask.sum(
            axis=1, dtype=np.uint64
        )

    for array, max_depth in max_depths.items():
        mask = row_counts[array][:max_depth] > 0
        rho_app_means[array][:max_depth][mask] = (
            row_sums[array][:max_depth][mask]
            / row_counts[array][:max_depth][mask]
        )
        rho_app_sq_sums[array][:max_depth][mask] = \
            row_sums_sq[array][:max_depth][mask]
        num_pixels[array][:max_depth] = \
            row_counts[array][:max_depth]

    return (
        array_types,
        depth_levels,
        rho_app_means,
        rho_app_sq_sums,
        rho_app_maxs,
        rho_app_mins,
        num_pixels,
    )


def giga_flatten(dataloader: DataLoader,
                 nb_pixels: dict[str, npt.NDArray[np.uint64]],
                 max_depth: dict[str, int],
                 lmdb_path: Path
                 ):
    flattened: dict[str, list[list[npt.NDArray]]] = {
        array: [[] for _ in range(max_depth[array])]
        for array in nb_pixels
    }
    for batch in tqdm(dataloader, total=len(dataloader), unit="batch", desc="Flattening batches"):
        for idx, pseudosection in enumerate(batch[3]):
            array = batch[2][idx]
            for row_idx, row in enumerate(pseudosection):
                valid_mask = ~np.isnan(row)
                flattened[array][row_idx].append(row[valid_mask])

    save_path = lmdb_path / "flattened"
    save_path.mkdir(exist_ok=True, parents=True)

    bins = 50
    # Save histograms for each depth level
    for array, max_depth in max_depth.items():
        all_data = np.concatenate([np.concatenate(flattened[array][row_idx]) for row_idx in range(max_depth) if flattened[array][row_idx]])

        # Get global x-axis range
        global_xmin, global_xmax = all_data.min(), all_data.max()

        # Get global y-axis max (density normalization)
        global_max_count = 0
        for row_idx in range(max_depth):
            if flattened[array][row_idx]:  # Avoid empty levels
                counts, _ = np.histogram(np.concatenate(flattened[array][row_idx]), bins=bins, density=True)
                global_max_count = max(global_max_count, counts.max())

        for row_idx in tqdm(reversed(range(max_depth)), total=max_depth, unit="depth level", desc=f"Saving histograms for {array}"):
            data = np.concatenate(flattened[array][row_idx])
            plt.hist(
                data,
                bins=bins,
                alpha=0.5,
                density=True,
                label=f"{array} - Depth level {row_idx}"
            )
            plt.legend(loc="upper right")
            plt.xlabel("Apparent resistivity")
            plt.ylabel("Frequency")
            plt.title(f"Apparent resistivity distribution for {array} DP: {row_idx}")

            plt.xlim(global_xmin, global_xmax)
            plt.ylim(0, global_max_count)

            plt.savefig(save_path / f"app_res_{array}_DP_{row_idx}.png")
            plt.close()

def compute_stats(arr: np.ndarray
                  ) -> tuple[np.int32, np.int32, np.float64, np.float64]:
    """
    Compute min, max, mean, and standard deviation of the given array.

    Parameters
    ----------
    arr: np.ndarray
        Numpy array for which to compute statistics.

    Returns
    -------
    tuple[np.int32, np.int32, np.float64, np.float64]
        A tuple of (min, max, mean, std).
    """
    return arr.min(), arr.max(), arr.mean(), arr.std()


def compute_pseudo_section_stats(dataloader: DataLoader,
                                 dataset: LMDBDataset,
                                 lmdb_path: Path,
                                 plot: bool = False
                                 ) -> None:
    """
    Compute and log pseudo section statistics from the data batches.

    Parameters
    ----------
    dataloader: DataLoader
        DataLoader object for the dataset.
    """
    print("Computing pseudo section statistics...")

    dataset_size: int = dataset.length
    array_types: npt.NDArray[np.bool_] = np.empty(
        dataset_size, dtype=np.bool_
    )
    depth_levels: npt.NDArray[np.int32] = np.empty(
        dataset_size, dtype=np.int32
    )

    max_depths: dict[str, int] = {
        "wa": 31,
        "slm": 47
    }
    rho_app_means: dict[str, list[float]] = {
        array: np.zeros((max_depth,), dtype=np.float64)
        for array, max_depth in max_depths.items()
    }
    rho_app_sq_sums: dict[str, list[float]] = {
        array: np.zeros((max_depth,), dtype=np.float64)
        for array, max_depth in max_depths.items()
    }
    rho_app_maxs: dict[str, list[float]] = {
        array: np.zeros((max_depth,), dtype=np.float64)
        for array, max_depth in max_depths.items()
    }
    rho_app_mins: dict[str, list[float]] = {
        array: np.zeros((max_depth,), dtype=np.float64)
        for array, max_depth in max_depths.items()
    }
    num_pixels: dict[str, list[float]] = {
        array: np.zeros((max_depth,), dtype=np.uint64)
        for array, max_depth in max_depths.items()
    }

    for batch_idx, batch in tqdm(
        enumerate(dataloader), total=len(dataloader), unit="batch"
    ):
        print(f"type(batch): {type(batch)}")
        print(f"type(batch[0]): {type(batch[0])}")
        print(f"type(batch[1]): {type(batch[1])}")
        print(f"type(batch[2]): {type(batch[2])}")
        print(f"type(batch[3]): {type(batch[3])}")
        print(f"type(batch[4]): {type(batch[4])}")
        print(f"type(batch[0][0]): {type(batch[0][0])}")
        print(f"type(batch[1][0]): {type(batch[1][0])}")
        print(f"type(batch[2][0]): {type(batch[2][0])}")
        print(f"type(batch[3][0]): {type(batch[3][0])}")
        print(f"type(batch[4][0]): {type(batch[4][0])}")
        print(f"type(batch[3][0][0]): {type(batch[3][0][0])}")
        print(f"type(batch[4][0][0]): {type(batch[4][0][0])}")
        print(f"type(batch[3][0][0][0]): {type(batch[3][0][0][0])}")
        print(f"type(batch[4][0][0][0]): {type(batch[4][0][0][0])}")
        (
            array_type,
            depth_level,
            rho_app_means_batch,
            rho_app_sq_sum,
            rho_app_max,
            rho_app_min,
            num_pixels_batch
        ) = process_batch(batch)

        start = batch_idx * dataloader.batch_size
        end = start + len(array_type)
        array_types[start:end] = array_type
        depth_levels[start:end] = depth_level

        for array, num_pixel_batch in num_pixels_batch.items():
            min_size = min(
                len(rho_app_means[array]), len(rho_app_means_batch[array])
            )
            valid_rows = np.arange(min_size)[
                num_pixel_batch[:min_size] > 0
            ]
            rho_app_means[array][valid_rows] += (
                rho_app_means_batch[array][valid_rows]
                * num_pixel_batch[valid_rows]
            )
            rho_app_sq_sums[array][valid_rows] += \
                rho_app_sq_sum[array][valid_rows]
            rho_app_maxs[array][valid_rows] = np.maximum(
                rho_app_maxs[array][valid_rows],
                rho_app_max[array][valid_rows]
            )
            rho_app_mins[array][valid_rows] = np.minimum(
                rho_app_mins[array][valid_rows],
                rho_app_min[array][valid_rows]
            )
            num_pixels[array][valid_rows] += \
                num_pixel_batch[valid_rows]

    rho_app_stds: dict[str, npt.NDArray[np.float64]] = {
        array: np.zeros((max_depth,), dtype=np.float64)
        for array, max_depth in max_depths.items()
    }
    for array, num_pixel in num_pixels.items():
        valid_rows = num_pixel > 0
        rho_app_means[array][valid_rows] /= num_pixel[valid_rows]
        rho_app_sq_sums[array][valid_rows] /= num_pixel[valid_rows]
        rho_app_stds[array][valid_rows] = np.sqrt(
            rho_app_sq_sums[array][valid_rows]
            - rho_app_means[array][valid_rows] ** 2
        )

    # Séparation en fonction du type
    depth_levels_dict: dict[str, npt.NDArray[np.int32]] = {
        array: depth_levels[array_types == is_wa]
        for array, is_wa in zip(["wa", "slm"], [True, False])
    }

    print("Computing min, max, mean, and standard deviation statistics...")
    stats: dict[str, tuple[int, int, float, float]] = {
        array: compute_stats(depth_levels_dict[array])
        for array in depth_levels_dict
    }

    for array, stat in stats.items():
        print(f"\nArray type: {array}")
        print(
            f"\tMinimum number of depth levels: {stat[0]}, "
            f"maximum number of depth levels: {stat[1]}"
        )
        print(
            f"\tMean number of depth levels: {stat[2]:.2f}, "
            f"std of number of depth levels: {stat[3]:.2f}"
        )

    np.save(lmdb_path / "rho_app_means_wa.npy", rho_app_means["wa"])
    np.save(lmdb_path / "rho_app_means_slm.npy", rho_app_means["slm"])
    np.save(lmdb_path / "rho_app_maxs_wa.npy", rho_app_maxs["wa"])
    np.save(lmdb_path / "rho_app_maxs_slm.npy", rho_app_maxs["slm"])
    np.save(lmdb_path / "rho_app_mins_wa.npy", rho_app_mins["wa"])
    np.save(lmdb_path / "rho_app_mins_slm.npy", rho_app_mins["slm"])

    if plot:
        for array, stat in stats.items():
            plt.hist(
                depth_levels_dict[array],
                bins=np.arange(stat[0], stat[1] + 2),
                alpha=0.5,
                label=array
            )
        plt.legend(loc="upper right")
        plt.xlabel("Depth levels")
        plt.ylabel("Frequency")
        plt.title("Depth level distribution")
        plt.show()

        for array, max_depth in max_depths.items():
            plt.bar(range(max_depth), rho_app_means[array], alpha=0.5, label=array)

        plt.legend(loc="upper right")
        plt.xlabel("Depth levels")
        plt.ylabel("Mean apparent resistivity")
        plt.title("Mean apparent resistivity distribution")
        plt.show()

        for array, max_depth in max_depths.items():
            plt.bar(range(max_depth), rho_app_stds[array], alpha=0.5, label=array)
        plt.legend(loc="upper right")
        plt.xlabel("Depth levels")
        plt.ylabel("Standard deviation")
        plt.title("Standard deviation distribution")
        plt.show()

        for array, max_depth in max_depths.items():
            plt.bar(range(max_depth), num_pixels[array], alpha=0.5, label=array)
        plt.legend(loc="upper right")
        plt.xlabel("Depth levels")
        plt.ylabel("Number of pixels")
        plt.title("Number of pixels distribution")
        plt.show()

        for array, max_depth in max_depths.items():
            plt.bar(range(max_depth), rho_app_maxs[array], alpha=0.5, label=array)
        plt.legend(loc="upper right")
        plt.xlabel("Depth levels")
        plt.ylabel("Maximum apparent resistivity")
        plt.title("Maximum apparent resistivity distribution")
        plt.show()

        for array, max_depth in max_depths.items():
            plt.bar(range(max_depth), rho_app_mins[array], alpha=0.5, label=array)
        plt.legend(loc="upper right")
        plt.xlabel("Depth levels")
        plt.ylabel("Minimum apparent resistivity")
        plt.title("Minimum apparent resistivity distribution")
        plt.show()
    
    plt.close("all")

    giga_flatten(dataloader, num_pixels, max_depths, lmdb_path)


def main(lmdb_path: Path, plot: bool = False
         ) -> None:
    """
    Main function to load the dataset and compute pseudo section statistics.

    Parameters
    ----------
    lmdb_path: Path
        Path to the LMDB dataset.
    """
    dataset: LMDBDataset = LMDBDataset(lmdb_path)
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=8,
        collate_fn=lmdb_custom_collate_fn,
    )
    compute_pseudo_section_stats(dataloader, dataset, lmdb_path, plot)


def parse_args() -> Namespace:
    """
    Parse command-line arguments.

    Return
    ------
    args: Namespace
        Parsed command-line arguments.
    """
    parser: ArgumentParser = ArgumentParser(
        description="Compute pseudo section statistics from an LMDB dataset."
    )
    parser.add_argument(
        "lmdb_path", type=Path, help="Path to the LMDB dataset"
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot statistics"
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("invERT: Computing pseudo section statistics...")
    args = parse_args()
    main(args.lmdb_path, args.plot)
