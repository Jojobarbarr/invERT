from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree
import lmdb
import pickle
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "lmdb_dir",
        type=Path,
        help="Path to the LMDB directory to clean up.",
    )
    args = parser.parse_args()

    lmdb_dir: Path = args.lmdb_dir
    lmdb_new: Path = Path(str(lmdb_dir.parent) + "_filtered")
    lmdb_new.mkdir(exist_ok=True)

    index = 0
    with lmdb.open(str(lmdb_dir), readonly=True, lock=False) as src_env:
        total_entries = src_env.stat()["entries"]
        with lmdb.open(str(lmdb_new / "data.lmdb"), map_size=src_env.info()['map_size']) as dst_env:
            with src_env.begin(write=False) as src_txn, dst_env.begin(write=True) as dst_txn:
                cursor = src_txn.cursor()
                for _, value in tqdm(cursor, desc="Filtering dataset", unit="entry", total=total_entries):
                    values = pickle.loads(value)
                    if np.nanmin(values[3]) < 0:
                        continue
                    key = f"{index:08d}".encode('ascii')
                    index += 1
                    dst_txn.put(key, value)
    
    rmtree(lmdb_dir)