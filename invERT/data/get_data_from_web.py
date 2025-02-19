import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from tarfile import open as taropen
from tqdm import tqdm


def get_file_url_from_thredds(catalog_url: str,
                              model_tarname: str
                              ) -> str:
    """Fetch the correct tar file URL from the THREDDS catalog."""
    response: requests.Response = requests.get(catalog_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to access catalog: {catalog_url}")

    root: ET.Element = ET.fromstring(response.content)

    # Define the namespace
    namespace: dict[str, str] = {
        "ns": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
    }

    # Find all dataset elements
    datasets: list[ET.Element] = root.findall(
        ".//ns:dataset",
        namespaces=namespace
    )

    # Extract the URL path for each dataset
    available_files: list[str] = [dataset.attrib.get("urlPath", "")
                                  for dataset in datasets]

    for url_path in available_files:
        if model_tarname in url_path:  # Match the expected tar filename
            return f"https://thredds.nci.org.au/thredds/fileServer/{url_path}"

    raise ValueError(f"File {model_tarname} not found in THREDDS catalog.")


def download_tar(tar_url: str,
                 download_path: Path,
                 overwrite: bool = False,
                 ) -> Path | None:
    """Download a tar file."""
    print(f"Downloading tar file from {tar_url}.")

    # Create the download path if it does not exist
    try:
        download_path.mkdir(parents=True, exist_ok=False)
        print(f"Successfully created directory {download_path}.")
    except FileExistsError:
        if not overwrite:
            print(
                f"Directory {download_path} already exists. To overwrite "
                f"it, make sure to set the overwrite flag to True. If you "
                f"don't want to overwrite, please provide a different path."
            )
            return None
        else:
            print(f"Potentially overwriting directory {download_path} ...")

    # Download the tar file by chunks (large file)
    response: requests.Response = requests.get(tar_url, stream=True)
    response.raise_for_status()

    # Downloaded file name
    tar_filename: Path = download_path / Path(tar_url).name

    total_size = int(response.headers.get("content-length", 0))

    print(f"Downloading to {tar_filename} ...")

    # Download the tar file by chunks of chunk_size bytes
    with open(tar_filename, "wb") as file, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc="Downloading"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))
    return tar_filename


def extract_tar(
        tar_filename: Path,
) -> None:
    # Extract contents
    print(f"Extracting tar file {tar_filename} ...")
    with taropen(tar_filename, "r") as tar:
        members = tar.getmembers()
        for member in tqdm(tar,
                           desc="Extraction",
                           unit="file",
                           total=len(members)):
            tar.extract(member, path=tar_filename.parent)
    print(f"Successfully extracted tar file {tar_filename}.")


def delete_tar(
        tar_filename: Path,
) -> None:
    # Delete tar file
    print(f"Deleting tar file {tar_filename} ...")
    tar_filename.unlink()
    print(f"Successfully deleted tar file {tar_filename}.")


if __name__ == "__main__":
    EVENTS: dict[int, str] = {
        1: "FOLD",
        2: "FAULT",
        3: "UNCONFORMITY",
        4: "SHEAR-ZONE",
        5: "DYKE",
        6: "PLUG",
        7: "TILT",
    }

    # ##### YOU CAN EDIT THIS IF YOU NEED TO #####

    # The events lists allows to process any sub-set of the Noddyverse dataset.
    events_lists: list[list[int]] = [
        [i, j, k]
        for j in range(1, 8)
        for k in range(1, 8)
        for i in range(1, 8)
    ]
    # events_lists: list[list[int]] = [
    #     [2, 5, 7],
    #     [5, 2, 4],
    #     [7, 2, 2]
    # ]

    # Choose where you want to save the data.
    dataset_folder: Path = Path("../../../dataset/6")

    # If the dataserver changes, it happens here:
    catalog_url: str = (
        "https://thredds.nci.org.au/thredds/catalog/tm64/"
        "noddyverse/bulk_models/catalog.xml"
    )

    # ##### NOW BE CAREFUL IF YOU TOUCH ANYTHING ELSE #####

    for events_list in events_lists:
        assert len(events_list) == 3, "len(events_list) must be 3."
        events: str = (
            f"{EVENTS[events_list[0]]}_"
            f"{EVENTS[events_list[1]]}_"
            f"{EVENTS[events_list[2]]}"
        )
        tar_name: str = f"{events}.tar"

        tar_url: str = get_file_url_from_thredds(catalog_url, tar_name)
        tar_filename: Path | None = download_tar(tar_url,
                                                 dataset_folder,
                                                 overwrite=True)
        extract_tar(dataset_folder / tar_name)
        delete_tar(dataset_folder / tar_name)
        filepath = Path(dataset_folder) / "models_by_code/models" / events
        files = filepath.glob("*")
        for file in files:
            if not file.suffixes == [".g12", ".gz"]:
                file.unlink()
