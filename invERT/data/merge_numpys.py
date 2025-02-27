import numpy as np
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser, Namespace


def merge_samples(input_dir: Path,
                  output_dir: Path,
                  combined_size: int = 8192
                  ) -> None:
    """
    Merges .npz files found recursively in input_dir into larger samples.
    Each merged file contains up to 'combined_size' arrays of shape (200, 200).
    """
    output_dir.mkdir(parents=True,
                     exist_ok=True)
    global_array: np.ndarray[np.int8] = np.zeros((combined_size, 200, 200),
                                                 dtype=np.int8)
    idx: int = 0  # current position in global_array
    counter: int = 0  # merged file counter

    folders: list[Path] = sorted(input_dir.glob("*"))
    for folder in tqdm(folders, desc="Merging npz files", unit="folder"):
        # Skip the folder with "SHEAR-ZONE", "DYKE", "PLUG" or "TILT" in the
        # name
        if any(
            substring in folder.name for substring in [
                "SHEAR-ZONE",
                "DYKE",
                "PLUG",
                "TILT"]):
            print(f"Skipping {folder.name}")
            continue
        # Retrieve all .npz files in input_dir (including subdirectories)
        files: list[Path] = sorted(folder.rglob("*.npz"))
        for file in files:
            data: np.ndarray[np.int8] = np.load(file)["arr_0"]
            data_len: int = len(data)
            start: int = 0
            # If data does not fit entirely in the remaining space,
            # write what you can, save the merged file, and then continue.
            while start < data_len:
                space_left: int = combined_size - idx
                chunk_size: int = min(space_left, data_len - start)
                global_array[idx: idx + chunk_size] = \
                    data[start: start + chunk_size]
                idx += chunk_size
                start += chunk_size
                if idx == combined_size:
                    np.savez_compressed(output_dir / f"{counter}.npz",
                                        global_array)
                    counter += 1
                    idx = 0

    # Save any leftover data that didn't fill an entire block
    if idx > 0:
        np.savez_compressed(output_dir / f"{counter}.npz", global_array[:idx])


def main(args: Namespace) -> None:
    if args.type == "all":
        input_dir = Path("../../../dataset/clean_reduced/")
        output_dir = Path("../../../dataset/clean_reduced_unified/")
    else:  # args.type == "sub"
        if not args.first_event:
            raise ValueError("first_event must be provided when type is 'sub'")
        input_dir = Path("../../../dataset") / args.first_event / \
            "models_by_code/models"
        output_dir = Path("../../../dataset/clean_reduced/") / args.first_event

    merge_samples(input_dir, output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "type",
        type=str,
        choices=["all", "sub"],
        help=("If 'all', merge all cleaned samples. If 'sub', merge only the "
              "sub samples for the specified first_event.")
    )
    parser.add_argument(
        "-f",
        "--first_event",
        type=str,
        help="Name of the first event (required when type is 'sub')"
    )
    args = parser.parse_args()
    main(args)
