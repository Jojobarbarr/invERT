import argparse
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import tarfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global namespace for the THREDDS catalog.
THREDDS_NAMESPACE = {"ns": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}


def get_catalog_xml(catalog_url: str) -> ET.Element:
    """Download and parse the catalog XML once."""
    response = requests.get(catalog_url)
    response.raise_for_status()
    return ET.fromstring(response.content)


def get_file_url_from_catalog(catalog_root: ET.Element,
                              model_tarname: str
                              ) -> str:
    """Extract the correct file URL from the cached catalog XML."""
    for dataset in catalog_root.findall(".//ns:dataset",
                                        namespaces=THREDDS_NAMESPACE):
        url_path = dataset.attrib.get("urlPath", "")
        if model_tarname in url_path:
            return f"https://thredds.nci.org.au/thredds/fileServer/{url_path}"
    raise ValueError(f"File {model_tarname} not found in THREDDS catalog.")


def download_tar(tar_url: str, download_path: Path) -> Path:
    """Download the tar file using streamed requests."""
    tar_filename = download_path / Path(tar_url).name

    
    if (dataset_folder / "models_by_code/models" / tar_filename.stem).exists():
        print(f"{tar_filename} already exists. Skipping download.")
        return tar_filename

    print(f"Downloading {tar_filename.name} from {tar_url} ...")
    with requests.get(tar_url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with tar_filename.open("wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {tar_filename.name}"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return tar_filename


def extract_tar(tar_filename: Path) -> None:
    """Extract the tar file with a progress bar."""
    print(f"Extracting {tar_filename.name} ...")
    with tarfile.open(tar_filename, "r:*") as tar:
        members = tar.getmembers()
        for member in tqdm(
            members,
            desc=f"Extracting {tar_filename.name}",
            unit="file"
        ):
            tar.extract(member, path=tar_filename.parent)
    print(f"Finished extracting {tar_filename.name}.")


def delete_tar(tar_filename: Path) -> None:
    """Delete the tar file after extraction."""
    if tar_filename.exists():
        tar_filename.unlink()
        print(f"Deleted {tar_filename.name}.")


def clean_extracted_files(dataset_folder: Path, events: str) -> None:
    """
    In the extracted directory, remove files that do not match the desired pattern.
    Here we keep only files ending with ".g12.gz".
    """
    extraction_path = dataset_folder / "models_by_code/models" / events
    for file in extraction_path.glob("*"):
        if not file.name.endswith(".g12.gz"):
            file.unlink()


def process_events(events_list: list[int], events_map: dict[int, str],
                   catalog_root: ET.Element, dataset_folder: Path) -> None:
    """Process one tar file: get URL, download, extract, delete tar, and clean up."""
    # Build the events string and tar file name.
    events = f"{events_map[events_list[0]]}_{events_map[events_list[1]]}_{events_map[events_list[2]]}"
    tar_name = f"{events}.tar"
    print(f"\n=== Processing {tar_name} ===")
    
    # Get the file URL from the cached catalog.
    tar_url = get_file_url_from_catalog(catalog_root, tar_name)
    
    # Download, extract, delete, and clean up.
    tar_filepath = download_tar(tar_url, dataset_folder)
    extract_tar(tar_filepath)
    delete_tar(tar_filepath)
    clean_extracted_files(dataset_folder, events)
    print(f"=== Finished processing {tar_name} ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download and extract Noddyverse data.")
    parser.add_argument("first_event", type=int, help="First event subset to download.")
    args = parser.parse_args()
    first_event = args.first_event

    # Mapping for event names.
    EVENTS: dict[int, str] = {
        1: "FOLD",
        2: "FAULT",
        3: "UNCONFORMITY",
        4: "SHEAR-ZONE",
        5: "DYKE",
        6: "PLUG",
        7: "TILT",
    }

    # Create the list of event combinations.
    events_lists: list[list[int]] = [
        [first_event, j, k]
        for j in range(1, 8)
        for k in range(1, 8)
    ]

    # Define the folder where data will be stored.
    dataset_folder: Path = Path(f"../../../dataset/{EVENTS[first_event]}")
    dataset_folder.mkdir(parents=True, exist_ok=True)

    # Download the catalog XML once.
    catalog_url: str = (
        "https://thredds.nci.org.au/thredds/catalog/tm64/noddyverse/bulk_models/catalog.xml"
    )
    catalog_root = get_catalog_xml(catalog_url)

    # Use a ThreadPoolExecutor to parallelize processing.
    max_workers = min(8, len(events_lists))  # You can adjust the number of workers as needed.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_events, events_list, EVENTS, catalog_root, dataset_folder)
            for events_list in events_lists
            if len(events_list) == 3
        ]
        # Optionally wait for all futures and handle exceptions.
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
