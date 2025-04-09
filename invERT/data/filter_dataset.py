import io
import pickle
from argparse import ArgumentParser
from pathlib import Path

import lmdb
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from data import InvERTSample


def is_valid_entry(pseudosection: npt.NDArray[np.float32]
                   ) -> bool:
    """
    Check if the pseudosection is valid (no negative in apparent resistivity values).

    Parameters
    ----------
    pseudosection : npt.NDArray[np.float32]
        The pseudosection to check.
    Returns
    -------
    bool
        True if the pseudosection is valid, False otherwise.
    """
    return np.nanmin(pseudosection) >= 0


def to_tensor_sample(values: tuple[int, int, str, npt.NDArray[np.float32], npt.NDArray[np.float32]]
                     ) -> InvERTSample:
    """
    Convert raw values to an InvERTSample with torch Tensors.

    Parameters
    ----------
    values : tuple[int, int, str, npt.NDArray[np.float32], npt.NDArray[np.float32]]
        The raw values to convert.
    Returns
    -------
    InvERTSample
        The converted InvERTSample.
    """
    num_electrode, subsection_length, array_type, pseudosection, norm_log_resistivity_model = values

    pseudosection = np.nan_to_num(pseudosection, nan=0.0)

    return InvERTSample(
        num_electrode=torch.tensor(num_electrode, dtype=torch.int32),
        subsection_length=torch.tensor(subsection_length, dtype=torch.int32),
        array_type=torch.tensor([1, 0] if array_type == "wa" else [0, 1], dtype=torch.int32),
        pseudosection=torch.tensor(pseudosection, dtype=torch.float32),
        norm_log_resistivity_model=torch.tensor(norm_log_resistivity_model, dtype=torch.float32),
    )


def main(lmdb_dir: Path,
         output_dir: Path,
         batch_size: int,
         ) -> None:
    output_dir.mkdir(exist_ok=True)

    index: int = 0

    with lmdb.open(str(lmdb_dir), readonly=True, lock=False) as src_env:
        total_entries: int = src_env.stat()["entries"]
        dst_path: str = str(output_dir / "data.lmdb")

        with lmdb.open(dst_path, map_size=2 ** 35, writemap=True, map_async=True) as dst_env:
            with src_env.begin(write=False) as src_txn:
                cursor: lmdb.Cursor = src_txn.cursor()

                write_txn: lmdb.Transaction = dst_env.begin(write=True)

                for i, (_, value) in enumerate(tqdm(cursor.iternext(values=True), desc="Filtering", total=total_entries, unit="entries")):
                    values: tuple[int, int, str, npt.NDArray[np.float32], npt.NDArray[np.float32]] = pickle.loads(value)
                    pseudosection: npt.NDArray[np.float32] = values[3]

                    if not is_valid_entry(pseudosection):
                        continue

                    sample: InvERTSample = to_tensor_sample(values)

                    
                    buffer: io.BytesIO = io.BytesIO()
                    torch.save(sample, buffer)
                    key = f"{index:08d}".encode("ascii")
                    write_txn.put(key, buffer.getvalue())
                    index += 1

                    if index % batch_size == 0:
                        write_txn.commit()
                        write_txn = dst_env.begin(write=True)
                
                if write_txn:
                    write_txn.commit()

    print(f"✅ Saved {index} valid samples to: {dst_path}")

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "lmdb_dir",
        type=Path,
        help="Path to the LMDB directory to clean up.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to the output directory for the filtered LMDB.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for LMDB writing.",
    )
    args = parser.parse_args()

    main(args.lmdb_dir, args.output_dir, args.batch_size)