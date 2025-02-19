import numpy as np
from pathlib import Path
import gzip
import tqdm
from typing import Tuple
from argparse import ArgumentParser


def remove_all_suffixes(path: Path) -> Path:
    while path.suffix:
        path = path.with_suffix("")
    return path


def save_array(npz_file_counter: int,
               combined_array: np.ndarray,
               files_to_supress: list[Path],
               ) -> Tuple[int, np.ndarray]:
    np.savez_compressed(data_path / f"{npz_file_counter}.npz", combined_array)
    print(f"\nSaved {npz_file_counter}.npz.\n")
    for file in files_to_supress:
        file.unlink()
    npz_file_counter += 1
    # Return a new empty buffer with the same dimensions.
    return (
        npz_file_counter, np.empty_like(combined_array, dtype=np.float64), []
    )


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser("Extract data from g12.gz files.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Pattern to match the folders."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("../../../dataset/6/models_by_code/models/"),
        help="Root folder of the data."
    )
    args = parser.parse_args()
    pattern: str = args.pattern
    data_root: Path = args.data_root
    total_nbr_folder = len([_ for _ in data_root.glob(pattern)])
    for folder_idx, folder in enumerate(data_root.glob(pattern), start=1):
        print(f"Processing {folder} ({folder_idx}/{total_nbr_folder})...")
        data_path: Path = folder
        files_to_extract = data_path.glob("*.gz")
        nbr_files_to_extract = len([_ for _ in data_path.glob("*.gz")])
        print(f"{nbr_files_to_extract} files in {data_path}.")

        indexes_list: list[int] = [0, 100, 199]
        buffer_size: int = 4000

        counter: int = 0
        npz_file_counter: int = 0

        combined_array: np.ndarray = np.empty((buffer_size, 200, 200),
                                              dtype=np.int8)
        files_to_supress: list[Path] = []
        for file in tqdm.tqdm(files_to_extract, desc="Extraction",
                              unit="file",
                              total=nbr_files_to_extract):
            files_to_supress.append(file)
            with gzip.open(file, 'rb') as gz_file:
                g12_array = np.loadtxt(gz_file)
            # Reformat the array to have (z, x, y) array.
            g12_array = g12_array.reshape((200, 200, 200))
            for array_sample, index in enumerate(indexes_list):
                if counter + array_sample == buffer_size:
                    npz_file_counter, combined_array, files_to_supress = \
                        save_array(
                            npz_file_counter,
                            combined_array,
                            files_to_supress
                        )
                    counter = 0
                combined_array[counter + array_sample] = g12_array[:, :, index]
            counter += len(indexes_list)
            for array_sample, index in enumerate(indexes_list):
                if counter + array_sample == buffer_size:
                    npz_file_counter, combined_array, files_to_supress = \
                        save_array(
                            npz_file_counter,
                            combined_array,
                            files_to_supress
                        )
                    counter = 0
                combined_array[counter + array_sample] = g12_array[:, index, :]
            counter += len(indexes_list)
        if nbr_files_to_extract != 0:
            _ = save_array(npz_file_counter, combined_array, files_to_supress)
        print(f"Finished processing {folder}.")
