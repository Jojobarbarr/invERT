import numpy as np
from pathlib import Path
import torch
import math
import random as rd
import pygimli as pg
from tqdm import tqdm
import pygimli.physics.ert as ert
import concurrent.futures
from argparse import ArgumentParser, Namespace


def detransform(log_res: float | np.ndarray[float]
                ) -> float | np.ndarray[float]:
    return 2 * 10 ** (4 * log_res)


def compute_active_columns(row: int,
                           is_even_row: bool,
                           total_cols: int,
                           offset: int
                           ) -> tuple[int, int]:
    """
    Compute the starting and ending column indices for the given row.
    """
    if is_even_row:
        col_start = math.ceil(row * 1.5) - offset
        col_end = total_cols - math.ceil(row * 1.5) + offset
    else:
        col_start = math.ceil((row + 1) * 1.5) - 1 - offset
        col_end = total_cols - math.ceil((row + 1) * 1.5) + 1 + offset
    return col_start, col_end


def define_electrodes_param(section_width: int
                            ) -> tuple[int, int, int]:
    nbr_electrodes: int = rd.randint(24, 96)
    # We keep 2 or 3 pixels between each electrodes
    if nbr_electrodes * 3 < section_width:
        space_between_electrodes: int = rd.randint(2, 3)
        total_pixels_to_keep: int = (nbr_electrodes - 1) * 3
    else:
        space_between_electrodes: int = 2
        total_pixels_to_keep: int = (nbr_electrodes - 1) * 2
    return nbr_electrodes, total_pixels_to_keep, space_between_electrodes


def generate_target(total_pixels_to_keep: int,
                    section: np.ndarray[np.int8]
                    ) -> np.ndarray[np.int8]:
    surface_level: int = rd.randint(
        0,
        section.shape[0] - total_pixels_to_keep // 2 - 1
    )
    left_border: int = rd.randint(
        0,
        section.shape[1] - total_pixels_to_keep - 1
    )

    depth_slice: slice = slice(
        surface_level,
        surface_level + total_pixels_to_keep // 2
    )
    width_slice: slice = slice(
        left_border,
        left_border + total_pixels_to_keep
    )

    return section[depth_slice, width_slice]


def generate_resistivity(target: np.ndarray[np.int8]
                         ) -> tuple[np.ndarray[np.float64],
                                    np.ndarray[np.float64]]:
    unique_classes, inv = np.unique(target, return_inverse=True)
    # Generate a random value for each unique rock class
    random_log_res = np.random.uniform(0, 1, size=len(unique_classes))
    # Use inverse indices to quickly assign values
    target_log_res = random_log_res[inv].reshape(target.shape)
    return target_log_res, detransform(target_log_res)


def generate_world(total_pixels_to_keep: int
                   ) -> pg.core.Mesh:
    x_arr: np.ndarray[np.float64] = np.linspace(
        0.,
        total_pixels_to_keep,
        total_pixels_to_keep + 1,
        dtype=np.float64
    )
    y_arr: np.ndarray[np.float64] = np.linspace(
        -(total_pixels_to_keep // 2),
        0,
        total_pixels_to_keep // 2 + 1,
        dtype=np.float64
    )
    return pg.createGrid(x=x_arr, y=y_arr, worldBoundaryMarker=True)


def generate_electrode_array(total_pixels_to_keep: int,
                             nbr_electrodes: int,
                             scheme_names: dict[str, str],
                             ) -> tuple[pg.DataContainerERT, str]:
    elec_array: np.float64 = np.linspace(
        0.,
        total_pixels_to_keep,
        nbr_electrodes,
        dtype=np.float64  # np.float64 to be compatible with C++ double
    )
    scheme_name = rd.choice(list(scheme_names.keys()))
    return ert.createData(elecs=elec_array,
                          schemeName=scheme_name), scheme_name


def process_pseudo_section_wenner_array(rhoa: list[float],
                                        nbr_electrodes: int
                                        ) -> np.ndarray[np.float64]:
    num_rows: int = (nbr_electrodes - 1) // 3
    num_cols: int = nbr_electrodes - 3

    even_num_cols: bool = (num_cols % 2 == 0)
    if even_num_cols:
        # We want non even number of columns to be able to center the
        # triangle.
        num_cols += 1

    offset: int = (nbr_electrodes - 1) % 2

    pseudo_section: np.ndarray[np.float64] = np.zeros(
        (num_rows, num_cols), dtype=np.float64)
    value_index: int = 0

    for i in range(num_rows):
        # Determine if the current row is considered "even" based on num_cols
        # parity
        is_even_row = (i % 2 == 0) if even_num_cols else (i % 2 == 1)

        col_start, col_end = compute_active_columns(
            i, is_even_row, num_cols, offset)

        for j in range(col_start, col_end):
            # For even rows, use a special average at the center column
            if is_even_row and j == (num_cols - 1) // 2:
                pseudo_section[i, j] = (
                    rhoa[value_index - 1] + rhoa[value_index]) / 2
            else:
                pseudo_section[i, j] = rhoa[value_index]
                value_index += 1
    return pseudo_section


def process_pseudo_section_schlumberger_array(rhoa: list[float],
                                              nbr_electrodes: int
                                              ) -> np.ndarray[np.float64]:
    num_cols: int = nbr_electrodes - 3
    num_lines: int = nbr_electrodes // 2 - 1

    pseudo_section: np.ndarray[np.float64] = np.zeros(
        (num_lines, num_cols), dtype=np.float64)

    value_index: int = 0
    for i in range(num_lines):
        start_col: int = i
        end_col: int = num_cols - i
        num_values_this_row: int = end_col - start_col
        pseudo_section[i, start_col:end_col] = rhoa[value_index: value_index +
                                                    num_values_this_row]
        value_index += num_values_this_row

    return pseudo_section


def save_sample_pt(filepath: Path,
                   samples: list[tuple[int, int, str, np.ndarray, np.ndarray]]
                   ) -> None:
    torch.save(samples, filepath)
    print(f"\nSaved {filepath.name}.\n")


def process_sample(section: np.ndarray[np.int8],
                   scheme_names: dict
                   ) -> tuple[int,
                              int,
                              str,
                              np.ndarray[np.float64],
                              np.ndarray[np.float64]
                              ]:
    nbr_electrodes, total_pixels_to_keep, space_between_electrodes = \
        define_electrodes_param(section.shape[1])

    target: np.ndarray = generate_target(total_pixels_to_keep, section)
    target_log_res, target_res = generate_resistivity(target)

    world: pg.core.Mesh = generate_world(total_pixels_to_keep)
    scheme, scheme_name = generate_electrode_array(
        total_pixels_to_keep,
        nbr_electrodes,
        scheme_names,
    )

    result: pg.DataContainerERT = ert.simulate(
        world,
        res=target_res.ravel(),
        scheme=scheme,
        verbose=False,
    )

    if scheme_name == "wa":
        pseudo_section: np.ndarray[np.float64] = \
            process_pseudo_section_wenner_array(result['rhoa'],
                                                nbr_electrodes)
    else:
        pseudo_section: np.ndarray[np.float64] = \
            process_pseudo_section_schlumberger_array(result['rhoa'],
                                                      nbr_electrodes)

    return (
        nbr_electrodes,
        space_between_electrodes,
        scheme_name,
        target_log_res,
        pseudo_section
    )


def count_samples(file):
    with np.load(file, mmap_mode='r') as data:
        return data["arr_0"].shape[0]


def flag_files_processed(output_path: Path,
                         nb_files_to_process: int
                         ) -> int:
    max_npz_processed = 0
    if not output_path.exists():
        print(f"Creating {output_path}...")
        output_path.mkdir(parents=True, exist_ok=True)
    for file in output_path.glob("*.txt"):
        max_npz_processed = max(max_npz_processed, int(file.stem))
    with open(output_path / f"{max_npz_processed + nb_files_to_process}.txt",
              "w") as f:
        f.write("")
    return max_npz_processed


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        "Generate synthetic samples from the given dataset.")
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=Path,
        default=Path("path/to/dataset"),
        help="Path to the dataset folder."
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        default=Path("../where/to/store/samples"),
        help="Path to the output folder."
    )
    parser.add_argument(
        "-n",
        "--number_of_file_to_process",
        type=int,
        default=2,
        help="Number of files to process."
    )
    parser.add_argument(
        "-np",
        "--non_parallel",
        action="store_true",
        help="Don't use parallel processing."
    )
    args: Namespace = parser.parse_args()

    dataset_path: Path = args.dataset_path

    output_path: Path = args.output_path

    number_of_file_to_process: int = args.number_of_file_to_process

    flag_files: int = flag_files_processed(output_path,
                                           number_of_file_to_process)
    files_to_process: list[Path] = [
        dataset_path / f"{idx}.npz"
        for idx in range(flag_files, flag_files + number_of_file_to_process)
    ]

    output_path.mkdir(parents=True, exist_ok=True)

    scheme_names: dict[str, str] = {
        "wa": "Wenner array",
        "slm": "Schlumberger array"
    }
    PARALLEL: bool = not args.non_parallel
    if PARALLEL:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(count_samples, files_to_process),
                                total=len(files_to_process),
                                desc="Counting samples",
                                unit="file"))
    else:
        results = [count_samples(file) for file in files_to_process]
    N_SAMPLES = sum(results)

    samples: list = []
    counter: int = 2 * flag_files

    for file_idx, file in enumerate(files_to_process):
        print(
            f"Processing {file.name} ({file_idx}/{len(files_to_process)})..."
        )
        multi_arrays = np.load(file)["arr_0"]

        if PARALLEL:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_sample,
                                           section,
                                           scheme_names)
                           for section in multi_arrays]

                for future in tqdm(concurrent.futures.as_completed(futures),
                                   total=len(multi_arrays),
                                   desc="Processing samples",
                                   unit="sample"):

                    samples.append(future.result())
                    if len(samples) >= 1024:
                        save_sample_pt(output_path / f"{counter}.pt", samples)
                        counter += 1
                        samples.clear()
        else:
            for section in tqdm(multi_arrays, desc="Processing samples",
                                unit="sample"):
                samples.append(process_sample(section, scheme_names))
                if len(samples) >= 1024:
                    save_sample_pt(output_path / f"{counter}.pt", samples)
                    counter += 1
                    samples.clear()
    if len(samples) > 0:
        save_sample_pt(output_path / f"{counter}.pt", samples)
