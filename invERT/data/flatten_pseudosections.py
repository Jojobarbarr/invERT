from argparse import ArgumentParser, Namespace
from pathlib import Path
import io
import lmdb
import torch
from tqdm import tqdm
from data import LMDBDataset
import shutil


def flatten(dataset: LMDBDataset,
            output_dir: Path,
            batch_size: int,
            ) -> None:
    index = 0
    with lmdb.open(str(output_dir), map_size=2 ** 35, readonly=False, writemap=True, map_async=True) as env:
        txn: lmdb.Transaction = env.begin(write=True)
        for sample in tqdm(dataset, desc=f'Computing min/max', unit='sample', total=len(dataset)):
            pseudosection = sample['pseudosection']
            sample['pseudosection'] = pseudosection[pseudosection != 0]
            buffer: io.BytesIO = io.BytesIO()
            torch.save(sample, buffer)
            key = f"{index:08d}".encode("ascii")
            txn.put(key, buffer.getvalue())
            index += 1

            if index % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)
        
        if txn:
            txn.commit()
if __name__ == "__main__":
    parser = ArgumentParser(description="Normalize the dataset")
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the dataset",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the LMDB database",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=65536,
        help="Batch size for processing samples",
    )
    args: Namespace = parser.parse_args()

    lmdb_path = Path(args.dataset_path)
    save_directory = Path(lmdb_path).parent.parent

    dataset = LMDBDataset(lmdb_path, readahead=True)

    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flatten(
        dataset,
        output_dir,
        args.batch_size,
    )
    print(f"Normalized dataset saved to {output_dir}")