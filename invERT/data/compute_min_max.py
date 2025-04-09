from argparse import ArgumentParser, Namespace
from pathlib import Path
import io
import lmdb
import torch
from tqdm import tqdm
from data import LMDBDataset


def compute_min_max(dataset: LMDBDataset,
                    save_directory: Path,
                    characteristics: list[str]
                    ) -> None:
    """
    Compute the minimum and maximum values of characteristics in the dataset.
    
    Parameters
    ----------
    dataset : LMDBDataset
        The dataset to compute min/max values for.
    save_directory : Path
        Directory to save the min/max results.
    """
    count = torch.load(save_directory / 'count.pt')
    minimums: dict[str, torch.Tensor] = {
        characteristic: {
            array_type: torch.empty((array_count)) for array_type, array_count in enumerate(count)
        }
        for characteristic in characteristics
    }
    maximums: dict[str, torch.Tensor] = {
        characteristic: {
            array_type: torch.empty((array_count)) for array_type, array_count in enumerate(count)
        }
        for characteristic in characteristics
    }

    local_count = torch.zeros(2, dtype=torch.int64)
    with torch.no_grad():
        for sample in tqdm(dataset, desc=f'Computing min/max', unit='sample', total=len(dataset)):
            array_type = torch.argmax(sample['array_type']).item()
            for characteristic in characteristics:
                characteristic_value = sample[characteristic]

                minimums[characteristic][array_type][local_count[array_type]] = characteristic_value.min()
                maximums[characteristic][array_type][local_count[array_type]] = characteristic_value.max()
            local_count[array_type] += 1

        for characteristic in characteristics:
            for array_type in range(len(count)):
                minimums[characteristic][array_type] = minimums[characteristic][array_type].min()
                maximums[characteristic][array_type] = maximums[characteristic][array_type].max()

    torch.save(minimums, save_directory / 'minimums.pt')
    torch.save(maximums, save_directory / 'maximums.pt')

    print('Minimums:')
    for characteristic in characteristics:
        print(f'\t{characteristic}:')
        for array_type in range(len(count)):
            print(f'\t\t{'schlumberger' if array_type else 'wenner'}: {minimums[characteristic][array_type]}')
    print('Maximums:')
    for characteristic in characteristics:
        print(f'\t{characteristic}:')
        for array_type in range(len(count)):
            print(f'\t\t{'schlumberger' if array_type else 'wenner'}: {maximums[characteristic][array_type]}')


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute min/max of characteristics")
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the dataset",
    )
    parser.add_argument(
        "-c",
        "--characteristics",
        type=str,
        nargs='+',
        default=['num_electrode', 'subsection_length', 'pseudosection'],
    )
    args: Namespace = parser.parse_args()

    lmdb_path = Path(args.dataset_path)
    save_directory = Path(lmdb_path).parent.parent

    characteristics = args.characteristics

    dataset = LMDBDataset(lmdb_path, readahead=True)
    compute_min_max(dataset, save_directory, characteristics)
    dataset.close()

