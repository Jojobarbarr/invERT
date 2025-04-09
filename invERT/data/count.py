from torch.utils.data import DataLoader
from data import LMDBDataset
from pathlib import Path
import torch
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

def count(dataset: LMDBDataset,
          save_directory: Path) -> None:
    """
    Count the number of samples in each category in the dataset.
    
    Parameters
    ----------
    dataset : LMDBDataset
        The dataset to count samples in.
    save_directory : Path
        Directory to save the count results.
    """
    dataloader = DataLoader(dataset, batch_size=2048, num_workers=8, prefetch_factor=8, collate_fn=LMDBDataset.lmdb_collate_fn_per_cat)
    count = torch.zeros(2, dtype=torch.int64)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Computing min/max', unit='batch', total=len(dataloader)):
            count += batch['array_types'].sum(dim=0)

    torch.save(count, save_directory / 'count.pt')
    print(f"Count: {count}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Count the number of samples in each category")
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the dataset",
    )
    args: Namespace = parser.parse_args()

    lmdb_path = Path(args.dataset_path)
    save_directory = Path(lmdb_path).parent.parent

    dataset = LMDBDataset(lmdb_path)
    count(dataset, save_directory)
    dataset.close()