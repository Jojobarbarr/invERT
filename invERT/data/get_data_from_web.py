import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from tarfile import open as taropen
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

THREDDS_NAMESPACE = {
    "ns": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
}


def get_file_url_from_thredds(catalog_url: str,
                              model_tarname: str
                              ) -> str:
    """Fetch the correct tar file URL from the THREDDS catalog."""
    response: requests.Response = requests.get(catalog_url)
    response.raise_for_status()

    root: ET.Element = ET.fromstring(response.content)

    datasets: list[ET.Element] = root.findall(
        ".//ns:dataset",
        namespaces=THREDDS_NAMESPACE
    )

    for dataset in datasets:
        url_path: str = dataset.attrib.get("urlPath", "")
        if model_tarname in url_path:
            return f"https://thredds.nci.org.au/thredds/fileServer/{url_path}"
    raise ValueError(f"File {model_tarname} not found in THREDDS catalog.")


def download_tar(tar_url: str,
                 download_path: Path,
                 ) -> Path | None:
    """Download a tar file, unless it already exists."""
    tar_filename: Path = download_path / Path(tar_url).name

    if tar_filename.exists():
        print(f"{tar_filename} already exists. Skipping download.")
        return tar_filename

    print(f"Downloading tar file from {tar_url} to {tar_filename}...")

    with requests.get(tar_url, stream=True) as response:
        response.raise_for_status()
        total_size: int = int(response.headers.get("content-length", 0))
        with tar_filename.open("wb") as file, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
    return tar_filename


def extract_tar(tar_filename: Path) -> None:
    """Extract tar file with progress feedback."""
    print(f"Extracting {tar_filename} ...")
    with taropen(tar_filename, "r:*") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting", unit="file"):
            tar.extract(member, path=tar_filename.parent)
    print(f"Successfully extracted {tar_filename}.")


def delete_tar(tar_filename: Path) -> None:
    """Delete the tar file."""
    if tar_filename.exists():
        tar_filename.unlink()
        print(f"Deleted {tar_filename}.")


def process_events(events_list: list[int],
                   events_map: dict[int, str],
                   catalog_url: str,
                   dataset_folder: Path
                   ) -> None:
    """Process a single events list: download, extract, and clean up."""
    # Construct the events name and tar filename.
    events: str = (f"{events_map[events_list[0]]}_"
                   f"{events_map[events_list[1]]}_"
                   f"{events_map[events_list[2]]}")
    tar_name: str = f"{events}.tar"

    # Get the file URL and download the tar.
    tar_url: str = get_file_url_from_thredds(catalog_url, tar_name)
    tar_filepath: Path | None = download_tar(tar_url, dataset_folder)

    # Extract and delete the tar file.
    extract_tar(tar_filepath)
    delete_tar(tar_filepath)

    # Clean up extracted files:
    # Keep only files ending with ".g12.gz" and remove others.
    extraction_path: Path = dataset_folder / "models_by_code/models" / events
    for file in extraction_path.glob("*"):
        if not file.name.endswith(".g12.gz"):
            file.unlink()


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        "Download and extract Noddyverse data."
    )
    parser.add_argument("first_event",
                        type=int,
                        help="First event subset to download.")
    args: Namespace = parser.parse_args()
    first_event: int = args.first_event

    EVENTS: dict[int, str] = {
        1: "FOLD",
        2: "FAULT",
        3: "UNCONFORMITY",
        4: "SHEAR-ZONE",
        5: "DYKE",
        6: "PLUG",
        7: "TILT",
    }

    # Generate the list of event combinations.
    events_lists: list[list[int]] = [
        [first_event, j, k]
        for j in range(1, 8)
        for k in range(1, 8)
    ]

    # Create the dataset folder once.
    dataset_folder: Path = Path(f"../../../dataset/{EVENTS[first_event]}")
    dataset_folder.mkdir(parents=True, exist_ok=True)

    # Catalog URL (dataserver URL can be changed here if needed)
    catalog_url: str = (
        "https://thredds.nci.org.au/thredds/catalog/"
        "tm64/noddyverse/bulk_models/catalog.xml"
    )

    # Process each events list.
    for events_list in events_lists:
        if len(events_list) != 3:
            continue  # safety check
        process_events(events_list, EVENTS, catalog_url, dataset_folder)
