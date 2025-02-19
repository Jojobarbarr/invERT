import numpy as np
from pathlib import Path
import gzip
import tqdm
from typing import Tuple, List
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor


def process_file(file: Path) -> Tuple[Path, List[np.ndarray]]:
    """
    Open a gzip file, load its text data, reshape to (200,200,200),
    and extract four 2D slices (three from axis=2 and three from axis=1).
    """
    with gzip.open(file, 'rb') as gz_file:
        g12_array: np.ndarray[np.int8] = np.loadtxt(gz_file, dtype=np.int8)
    g12_array = g12_array.reshape((200, 200, 200))
    indexes_list: list[int] = [66, 132]
    # Extract slices along the third axis.
    slices: list[slice] = [g12_array[:, :, idx] for idx in indexes_list]
    # Extract slices along the second axis.
    slices += [g12_array[:, idx, :] for idx in indexes_list]
    return file, slices


def flush_buffer(npz_file_counter: int,
                 combined_array: np.ndarray,
                 count: int,
                 files_to_supress: List[Path],
                 data_path: Path
                 ) -> Tuple[int, int, List[Path]]:
    """
    Save the filled portion of the buffer to disk and delete the contributing
    gzip files.
    """
    np.savez_compressed(
        data_path / f"{npz_file_counter}.npz",
        combined_array[:count])
    print(f"\nSaved {npz_file_counter}.npz.\n")
    for file in files_to_supress:
        file.unlink()
    npz_file_counter += 1
    # Reset counter and file list.
    return npz_file_counter, 0, []


if __name__ == "__main__":
    parser = ArgumentParser("Extract data from g12.gz files concurrently.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_FOLD_*",
        help="Pattern to match the folders."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("../../../dataset/5/models_by_code/"),
        help="Root folder of the data."
    )
    args = parser.parse_args()

    pattern: str = args.pattern
    data_root: Path = args.data_root

    buffer_size: int = 2048

    total_nbr_folder: int = len(list(data_root.glob(pattern)))
    for folder_idx, folder in enumerate(data_root.glob(pattern), start=1):
        print(f"Processing {folder} ({folder_idx}/{total_nbr_folder})...")
        list_of_files: list[Path] = list(folder.glob("*.gz"))
        nbr_files_to_extract: int = len(list_of_files)
        print(f"{nbr_files_to_extract} files in {folder}.")

        if nbr_files_to_extract == 0:
            continue

        combined_array: np.ndarray[np.int8] = np.empty(
            (buffer_size, 200, 200),
            dtype=np.int8)
        counter: int = 0
        npz_file_counter: int = len(list(folder.glob("*.npz")))
        files_to_supress: List[Path] = []

        # Process files concurrently.
        with ProcessPoolExecutor() as executor:
            for file, slices in tqdm.tqdm(
                executor.map(process_file, list_of_files),
                total=nbr_files_to_extract,
                desc="Extraction",
                unit="file"
            ):
                files_to_supress.append(file)
                for slice_array in slices:
                    # If the buffer is full, flush it.
                    if counter == buffer_size:
                        npz_file_counter, counter, files_to_supress = \
                            flush_buffer(
                                npz_file_counter,
                                combined_array,
                                counter,
                                files_to_supress,
                                folder
                            )
                    combined_array[counter] = slice_array
                    counter += 1

        # Flush any remaining data in the buffer.
        if counter > 0:
            npz_file_counter, counter, files_to_supress = flush_buffer(
                npz_file_counter,
                combined_array,
                counter,
                files_to_supress,
                folder
            )
        print(f"Finished processing {folder}.")
