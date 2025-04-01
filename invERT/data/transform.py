import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from shutil import rmtree
from itertools import zip_longest

import numpy as np
import numpy.typing as npt
import torch.multiprocessing as mp
import lmdb
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import LMDBDataset, lmdb_custom_collate_fn, invERTbatch
import data_transforms as daT

mp.set_sharing_strategy('file_system')


def shift(dataloader: DataLoader,
          arrays: tuple[str]
          ) -> dict[str, npt.NDArray[np.float64]]:
    """
    Computes the minimum apparent resistivity values for each row for each array.
    
    Parameters
    ----------
    dataloader: DataLoader
        DataLoader containing the LMDB dataset.
    arrays: tuple[str]
        Tuple containing the array names.

    Returns
    -------
    rho_app_min: dict[str, npt.NDArray[np.float64]]
        Dictionary containing the minimum apparent resistivity values for each row
        for each array.
    """
    rho_app_mins: dict[str, list[npt.NDArray[np.float64]]] = {array: [] for array in arrays}
    for _, batch in tqdm(dataloader, desc="Computing minimum apparent resistivity", unit="batch"):
        for array in arrays:
            rho_app_min_batch = daT.compute_min(batch, arrays)
            rho_app_mins[array].append(rho_app_min_batch[array])
    rho_app_min: dict[str, npt.NDArray[np.float64]] = {
        array: np.array([min(mins) for mins in zip_longest(*rho_app_mins[array], fillvalue=np.inf)])
        for array in arrays
    }
    return rho_app_min


def write_to_lmdb(txn: lmdb.Transaction,
                  sample_indices: list[int],
                  batch: invERTbatch,
                  ) -> None:
    for sample_idx, values in zip(sample_indices, zip(*batch)):
        key = f"{sample_idx:08d}".encode("ascii")
        data = pickle.dumps(values)
        txn.put(key, data)
        del data


def pre_transfrom(dataloader: DataLoader,
                  lmdb_path: Path,
                  arrays: tuple[str],
                  args: Namespace,
                  ) -> None:
    """
    Pre-transforms the pseudo sections in the dataset.

    Parameters
    ----------
    dataloader: DataLoader
        DataLoader containing the LMDB dataset.
    lmdb_path: Path
        Path to the LMDB dataset.
    arrays: tuple[str]
        Tuple containing the array names.
    args: Namespace
        Parsed command-line arguments.
    
    Returns
    -------
    None
    """
    
    if args.copy or not (lmdb_path / 'data.lmdb').exists():
        if args.shift:
            rho_app_min = shift(dataloader, arrays)
        if (lmdb_path / 'data.lmdb').exists():
            rmtree(lmdb_path / 'data.lmdb')
        env = lmdb.open(str(lmdb_path / 'data.lmdb'), map_size= 2 ** 35)  # 32 GB
        description = f"{'Pre-transforming and w' if args.shift or args.log else 'W'}riting to LMDB copy"
        with env.begin(write=True) as txn:
            for sample_indices, batch in tqdm(dataloader, desc=description, unit="batch"):
                new_batch = batch
                if args.shift:
                    new_batch = daT.shift(new_batch, rho_app_min)
                if args.log:
                    new_batch = daT.log_transform(new_batch)
                write_to_lmdb(txn, sample_indices, new_batch)
        env.close()
    else:
        print(
            "Skipping pre-transformations as the processed LMDB dataset already exists.\n"
            "If you want to recompute the pre-transformations, use '-c' or '--copy'."
        )


def post_transfrom(dataloader: DataLoader,
                  lmdb_path: Path,
                  old_lmdb_path: Path,
                  flattened: dict[str, list[npt.NDArray]],
                  arrays: tuple[str],
                  args: Namespace,
                  ) -> None:
    """
    Pre-transforms the pseudo sections in the dataset.

    Parameters
    ----------
    dataloader: DataLoader
        DataLoader containing the LMDB dataset.
    lmdb_path: Path
        Path to the LMDB dataset.
    old_lmdb_path: Path
        Path to the old LMDB dataset.
    flattened: dict[str, list[npt.NDArray]]
        Dictionary containing the flattened pseudo sections for each array.
    arrays: tuple[str]
        Tuple containing the array names.
    args: Namespace
        Parsed command-line arguments.
    
    Returns
    -------
    None
    """
    if (lmdb_path / 'data.lmdb').exists():
        rmtree(lmdb_path / 'data.lmdb')
    max_value: dict[str, float] = {array: max(np.max(rho_app_row) for rho_app_row in flattened[array]) for array in arrays}
    env = lmdb.open(str(lmdb_path / 'data.lmdb'), map_size= 2 ** 35)  # 32 GB
    description = f"Post-transforming and writing to LMDB copy"
    with env.begin(write=True) as txn:
        for sample_indices, batch in tqdm(dataloader, desc=description, unit="batch"):
            new_batch = batch
            new_batch = daT.normalize(new_batch, max_value)
            write_to_lmdb(txn, sample_indices, new_batch)
    env.close()
    rmtree(old_lmdb_path / 'data.lmdb')


def plot_means(rho_app_means: dict[str, npt.NDArray[np.float64]],
               rho_app_stds: dict[str, npt.NDArray[np.float64]],
               arrays: tuple[str],
               lmdb_path: Path,
               args: Namespace
               ) -> None:
    """
    Plots the mean apparent resistivity values for each array.

    Parameters
    ----------
    rho_app_means: dict[str, npt.NDArray[np.float64]]
        Dictionary containing the mean apparent resistivity values for each array.
    rho_app_stds: dict[str, npt.NDArray[np.float64]]
        Dictionary containing the standard deviation apparent resistivity values for each array.
    arrays: tuple[str]
        Tuple containing the array names.
    lmdb_path: Path
        Path to the LMDB dataset.
    args: Namespace
        Parsed command-line arguments.

    Returns
    -------
    None
    """
    save_path = lmdb_path / "plot"
    save_path.mkdir(exist_ok=True)
    for arr_index, array in enumerate(arrays):
        x = range(len(rho_app_means[array]))
        y = rho_app_means[array]
        yerr = rho_app_stds[array]
        plt.bar(x, y, label=array, alpha=0.5, zorder=2)
        if args.log:
            plt.errorbar(x, y, yerr=yerr, fmt='none', elinewidth=1, capsize=2, capthick=1, zorder=3)
        plt.xlabel("Pseudodepth")
        plt.ylabel("Mean apparent resistivity")
        plt.title("Mean apparent resistivity and standard deviation for each pseudodepth")
        plt.legend()
        plt.grid(axis='y', linestyle="--", alpha=0.5)
        plt.savefig(save_path / f"mean_app_res_{array}{'_log' if args.log else ''}.png")
        plt.close()


def plot_hist_per_depth(flattened: dict[str, list[npt.NDArray]],
                        max_depth: dict[str, int],
                        arrays: tuple[str],
                        lmdb_path: Path,
                        args: Namespace
                        ) -> None:
    """
    Plots the histogram of the apparent resistivity values for each depth level.
    Parameters
    ----------
    flattened: dict[str, list[npt.NDArray]]
        Dictionary containing the flattened pseudo sections for each array.
    arrays: tuple[str]
        Tuple containing the array names.
    lmdb_path: Path
        Path to the LMDB dataset.
    args: Namespace
        Parsed command-line arguments.
    Returns
    -------
    None
    """
    num_bins = 100
    save_path = lmdb_path / "plot" / "histograms"
    save_path_zoom = save_path / "zoom"
    save_path.mkdir(exist_ok=True)
    save_path_zoom.mkdir(exist_ok=True)
    min_value: dict[str, float] = {array: min(np.min(rho_app_row) for rho_app_row in flattened[array]) for array in arrays}
    max_value: dict[str, float] = {array: max(np.max(rho_app_row) for rho_app_row in flattened[array]) for array in arrays}
    for array in arrays:
        array_data = flattened[array]
        num_levels = len(array_data)
        bins = np.linspace(min_value[array], max_value[array], num_bins + 1)
        bins_zoom = np.linspace(min_value[array], 10000, num_bins + 1)
        counts = [np.histogram(depth_level, bins=bins, density=True)[0] for depth_level in array_data]
        max_count = max(count.max() for count in counts)
        for depth_level in tqdm(range(num_levels), desc=f"Plotting histograms for {array}", unit="depth level"):
            plt.hist(
                array_data[depth_level],
                bins=bins,
                alpha=1,
                density=True,
                label=f"Depth level {depth_level}",
            )
            plt.xlabel("Apparent resistivity")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of apparent resistivity values for {array} at depth level {depth_level}")
            plt.legend()
            plt.grid(axis='y', linestyle="--", alpha=0.5)
            plt.xlim(min_value[array], max_value[array])
            plt.ylim(0, max_count)
            plt.savefig(save_path / f"histogram_{array}_depth_{depth_level}{'_log' if args.log else ''}.png")
            plt.close()

            plt.hist(
                array_data[depth_level],
                bins=bins_zoom,
                alpha=1,
                density=True,
                label=f"Depth level {depth_level}",
            )
            plt.xlabel("Apparent resistivity")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of apparent resistivity values for {array} at depth level {depth_level}")
            plt.legend()
            plt.grid(axis='y', linestyle="--", alpha=0.5)
            plt.xlim(min_value[array], 10000)
            plt.ylim(0, 0.0002)
            plt.savefig(save_path_zoom / f"histogram_{array}_depth_{depth_level}{'_log' if args.log else ''}.png")
            plt.close()




def process_data(dataloader: DataLoader,
                 arrays: tuple[str],
                 rho_app_sums: dict[str, list[npt.NDArray[np.float64]]],
                 rho_app_sums_squared: dict[str, list[npt.NDArray[np.float64]]],
                 rho_app_counts: dict[str, list[npt.NDArray[np.int64]]],
                 min_depths: dict[str, list[int]],
                 max_depths: dict[str, list[int]],
                 flattened: dict[str, list[npt.NDArray]]
                 ) -> tuple[
                     dict[str, list[npt.NDArray[np.float64]]],
                     dict[str, list[npt.NDArray[np.float64]]],
                     dict[str, list[npt.NDArray[np.int64]]],
                     dict[str, list[int]],
                     dict[str, list[int]],
                     dict[str, list[npt.NDArray]]
                 ]:
    """
    Processes the data to compute the pseudo section statistics.

    Parameters
    ----------
    dataloader: DataLoader
        DataLoader containing the LMDB dataset.
    arrays: tuple[str]
        Tuple containing the array names.
    rho_app_sums: dict[str, list[npt.NDArray[np.float64]]]
        Dictionary containing the sum of the apparent resistivity values for each array.
    rho_app_sums_squared: dict[str, list[npt.NDArray[np.float64]]]
        Dictionary containing the sum of the squared apparent resistivity values for each array.
    rho_app_counts: dict[str, list[npt.NDArray[np.int64]]]
        Dictionary containing the count of the apparent resistivity values for each array.
    min_depths: dict[str, list[int]]
        Dictionary containing the minimum depth for each array.
    max_depths: dict[str, list[int]]
        Dictionary containing the maximum depth for each array.
    flattened: dict[str, list[npt.NDArray]]
        Dictionary containing the flattened pseudo sections for each array.

    Returns
    -------
    tuple[
        dict[str, list[npt.NDArray[np.float64]]],
        dict[str, list[npt.NDArray[np.float64]]],
        dict[str, list[npt.NDArray[np.int64]]],
        dict[str, list[int]],
        dict[str, list[int]],
        dict[str, list[npt.NDArray]]
    ]
        Tuple containing the updated dictionaries
        for the sum of the apparent resistivity values,
        the sum of the squared apparent resistivity values,
        the count of the apparent resistivity values,
        the minimum depth, the maximum depth,
        and the flattened pseudo sections for each array.
    """
    for _, batch in tqdm(dataloader, desc="Computing statistics", unit="batch"):
        sums: tuple[
            list[npt.NDArray[np.float64]],
            list[npt.NDArray[np.float64]],
            list[npt.NDArray[np.int64]]
        ] = daT.compute_sum(batch)
        min_d: dict[str, int] = daT.compute_min_depth(batch, arrays)
        max_d: dict[str, int] = daT.compute_max_depth(batch, arrays)
        flatteneds: dict[str, list[npt.NDArray]] = daT.append_by_row(batch, arrays)
        array_b = batch[2]
        for array in arrays:
            mask = (np.array(array_b) == array)
            rho_app_sums[array].extend(sum_ for sum_, good in zip(sums[0], mask) if good)
            rho_app_sums_squared[array].extend(sum_squared for sum_squared, good in zip(sums[1], mask) if good)
            rho_app_counts[array].extend(count for count, good in zip(sums[2], mask) if good)
                                
            min_depths[array].append(min_d[array])
            max_depths[array].append(max_d[array])

            flattened[array] = [
                prev if actual is None else actual if prev is None
                else np.concatenate((prev, actual), axis=0)
                for prev, actual in zip_longest(flattened[array], flatteneds[array], fillvalue=None)
            ]
        
    return rho_app_sums, rho_app_sums_squared, rho_app_counts, min_depths, max_depths, flattened


def post_process_data(arrays: tuple[str],
                      rho_app_sums: dict[str, list[npt.NDArray[np.float64]]],
                      rho_app_sums_squared: dict[str, list[npt.NDArray[np.float64]]],
                      rho_app_counts: dict[str, list[npt.NDArray[np.int64]]],
                      min_depths: dict[str, list[int]],
                      max_depths: dict[str, list[int]]
                      ) -> tuple[
                          dict[str, npt.NDArray[np.float64]],
                          dict[str, npt.NDArray[np.float64]],
                          dict[str, npt.NDArray[np.int64]],
                          dict[str, int],
                          dict[str, int]
                      ]:
    """
    Post-processes the data to compute the pseudo section statistics.

    Parameters
    ----------
    arrays: tuple[str]
        Tuple containing the array names.
    rho_app_sums: dict[str, list[npt.NDArray[np.float64]]]
        Dictionary containing the sum of the apparent resistivity values for each array.
    rho_app_sums_squared: dict[str, list[npt.NDArray[np.float64]]]
        Dictionary containing the sum of the squared apparent resistivity values for each array.
    rho_app_counts: dict[str, list[npt.NDArray[np.int64]]]
        Dictionary containing the count of the apparent resistivity values for each array.
    min_depths: dict[str, list[int]]
        Dictionary containing the minimum depth for each array.
    max_depths: dict[str, list[int]]
        Dictionary containing the maximum depth for each array.

    Returns
    -------
    tuple[
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.int64]],
        dict[str, int],
        dict[str, int]
    ]
        Tuple containing the updated dictionaries
        for the sum of the apparent resistivity values,
        the sum of the squared apparent resistivity values,
        the count of the apparent resistivity values,
        the minimum depth, and the maximum
        depth for each array
    """
    min_depth: dict[str, int] = {array: min(min_depths[array]) for array in arrays}
    max_depth: dict[str, int] = {array: max(max_depths[array]) for array in arrays}

    rho_app_sum: dict[str, npt.NDArray[np.float64]] = {
        array: np.array(
            [
                sum(sums) for sums in tqdm(
                    zip_longest(*rho_app_sums[array], fillvalue=0),
                    desc=f"Summing {array}",
                    unit="depth level",
                    total=max_depth[array]
                )
            ]
        ) for array in arrays
    }
    rho_app_sum_squared: dict[str, npt.NDArray[np.float64]] = {
        array: np.array(
            [
                sum(sums) for sums in tqdm(
                    zip_longest(*rho_app_sums_squared[array], fillvalue=0),
                    desc=f"Summing (squared) {array}",
                    unit="depth level",
                    total=max_depth[array]
                )
            ]
        ) for array in arrays
    }
    rho_app_count: dict[str, npt.NDArray[np.int64]] = {
        array: np.array(
            [
                sum(counts) for counts in tqdm(
                    zip_longest(*rho_app_counts[array], fillvalue=0),
                    desc=f"Summing counts {array}",
                    unit="depth level",
                    total=max_depth[array]
                )
            ]
        ) for array in arrays
    }
    return rho_app_sum, rho_app_sum_squared, rho_app_count, min_depth, max_depth


def compute_stats(dataloader: DataLoader,
                  lmdb_path: Path,
                  args: Namespace
                  ) -> None:
    """
    Computes the pseudo section statistics.

    Parameters
    ----------
    dataloader: DataLoader
        DataLoader containing the LMDB dataset.
    args: Namespace
        Parsed command-line arguments.
    
    Returns
    -------
    None
    """
    arrays: tuple[str] = ("wa", "slm")

    if args.recompute or not (lmdb_path / "stats" / "rho_app_means.pkl").exists():
        rho_app_sums: dict[str, list[npt.NDArray[np.float64]]] = {array: [] for array in arrays}
        rho_app_sums_squared: dict[str, list[npt.NDArray[np.float64]]] = {array: [] for array in arrays}
        rho_app_counts: list[npt.NDArray[np.int64]] = {array: [] for array in arrays}
        min_depths: dict[str, list[int]] = {array: [] for array in arrays}
        max_depths: dict[str, list[int]] = {array: [] for array in arrays}
        flattened: dict[str, list[npt.NDArray]] = {
            array: [] for array in arrays
        }

        rho_app_sums, rho_app_sums_squared, rho_app_counts, min_depths, max_depths, flattened = \
            process_data(
                dataloader,
                arrays,
                rho_app_sums,
                rho_app_sums_squared,
                rho_app_counts,
                min_depths,
                max_depths,
                flattened
            )

        
        rho_app_sum, rho_app_sum_squared, rho_app_count, min_depth, max_depth = \
            post_process_data(
                arrays,
                rho_app_sums,
                rho_app_sums_squared,
                rho_app_counts,
                min_depths,
                max_depths
            )

        save_path = lmdb_path / "stats"
        save_path.mkdir(exist_ok=True)

        rho_app_means: dict[str, npt.NDArray[np.float64]] = {array: rho_app_sum[array] / rho_app_count[array] for array in arrays}
        rho_app_vars: dict[str, npt.NDArray[np.float64]] = {array: rho_app_sum_squared[array] / rho_app_count[array] - rho_app_means[array] ** 2 for array in arrays}
        rho_app_stds: dict[str, npt.NDArray[np.float64]] = {array: np.sqrt(rho_app_var) for array, rho_app_var in rho_app_vars.items()}

        pickle.dump(rho_app_means, open(save_path / "rho_app_means.pkl", "wb"))
        pickle.dump(rho_app_stds, open(save_path / "rho_app_stds.pkl", "wb"))
        pickle.dump(flattened, open(save_path / "flattened.pkl", "wb"))
        pickle.dump((min_depth, max_depth), open(save_path / "min_max_depth.pkl", "wb"))

    else:
        print(
            "Skipping statistics computation as the statistics already exist.\n"
            "If you want to recompute the statistics, use '-r' or '--recompute'."
        )
        rho_app_means = pickle.load(open(lmdb_path / "stats" / "rho_app_means.pkl", "rb"))
        rho_app_stds = pickle.load(open(lmdb_path / "stats" / "rho_app_stds.pkl", "rb"))
        flattened = pickle.load(open(lmdb_path / "stats" / "flattened.pkl", "rb"))
        min_depth, max_depth = pickle.load(open(lmdb_path / "stats" / "min_max_depth.pkl", "rb"))

    plot_means(rho_app_means, rho_app_stds, arrays, lmdb_path, args)
    plot_hist_per_depth(flattened, max_depth, arrays, lmdb_path, args)

    plt.close('all')

    final_data_path = lmdb_path.parent.parent / f"{lmdb_path.parent.stem}_post"
    final_data_path.mkdir(exist_ok=True)
    post_transfrom(
        dataloader,
        final_data_path,
        lmdb_path,
        flattened,
        arrays,
        args
    )

def main(args: Namespace,
         arrays: tuple[str]
         ) -> None:
    """
    Main function to load the dataset and compute pseudo section statistics.

    Parameters
    ----------
    args: Namespace
        Parsed command-line arguments.

    Returns
    -------
    None
    """
    lmdb_path = args.lmdb_path
    dataset: LMDBDataset = LMDBDataset(lmdb_path)

    
    proc_lmdb_path: Path = lmdb_path.parent.parent / f"{lmdb_path.parent.stem}_proc"
    proc_lmdb_path.mkdir(exist_ok=True)

    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=8,
        prefetch_factor=8,
        collate_fn=lmdb_custom_collate_fn,
    )
    pre_transfrom(dataloader, proc_lmdb_path, arrays, args)
    dataset.close()

    proc_lmdb_path /= "data.lmdb"
    dataset = LMDBDataset(proc_lmdb_path)
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,
        collate_fn=lmdb_custom_collate_fn,
    )
    compute_stats(dataloader, proc_lmdb_path, args)


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
        "-c",
        "--copy",
        action="store_true",
        help="Copy the LMDB dataset to a new location. This imply recomputing the pseudo section statistics. (-r)"
    )
    parser.add_argument(
        "-r",
        "--recompute",
        action="store_true",
        help="Recompute the pseudo section statistics."
    )
    parser.add_argument(
        "-s",
        "--shift",
        action="store_true",
        help="Shifts the pseudo sections."
    )
    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Log transforms the pseudo sections."
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot the pseudo section statistics."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.copy:
        args.recompute = True
    arrays: tuple[str] = ("wa", "slm")
    if args.log and not args.shift:
        print(
            "WARNING: Log transform will be applied without shifting the pseudo sections. "
            "If there are negative apparent resistivities values, the log transform will fail."
        )
    main(args, arrays)
