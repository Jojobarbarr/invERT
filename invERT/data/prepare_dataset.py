# SimPEG functionality
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps
from simpeg.utils.solver_utils import DefaultSolverWarning

# discretize functionality
from discretize import TensorMesh
from discretize.utils import active_from_xyz

import numpy as np
from pathlib import Path
import math
import lmdb

from pymatsolver import Pardiso

import random as rd
from argparse import ArgumentParser, Namespace
import concurrent.futures
import pickle
from os import cpu_count

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    # If tqdm is not installed, we define a dummy function that just returns
    # the iterable.
    def tqdm(iterable, *args, **kwargs):
        return iterable

# To suppress warnings DefaultSolverWarning from SimPEG
import warnings
warnings.filterwarnings(
    "ignore", category=DefaultSolverWarning
)


def create_slice(max_length: int,
                 fraction: float,
                 start: int | None = None,
                 ) -> slice:
    """
    Creates a slice of a given fraction of the total length.

    The slice can be randomly placed or start at a given index.

    Parameters
    ----------
    max_length : int
        The length of the sample we want to extract the slice from.
    fraction : float
        The fraction of the total length to keep.
    start : int, optional
        The starting index of the slice. If None, a random index is chosen.

    Returns
    -------
    slice
        The slice object.
    """
    if start is None:
        start: int = rd.randint(0, int(max_length * (1 - fraction)))
    else:
        start: int = start
    return slice(start, start + int(max_length * fraction))


def extract_subsection(
        section: np.ndarray[np.int8],
        subsection_length: int,
        vertical_fraction: float,
        start: tuple[int | None, int | None] | None = (None, None),
        return_slices: bool = False,
) -> np.ndarray[np.int8] | tuple[np.ndarray[np.int8], slice, slice]:
    """
    Extracts a subsection of the section.

    The subsection can be randomly placed in the section or start at a given
    index.

    Parameters
    ----------
    section : np.ndarray[np.int8]
        The section from which to extract the subsection.
    subsection_length : int
        The total number of pixels to keep in the subsection.
    vertical_fraction : float
        The fraction of the section to keep vertically.
    start : tuple[int | None, int | None], optional
        The starting index of the subsection. If None, a random index is
        chosen. Index 0 is the vertical index and index 1 is the horizontal
        index.
    return_slices : bool, optional
        If True, the function returns the subsection and the slices used to
        extract it.

    Returns
    -------
    np.ndarray[np.int8] or tuple[np.ndarray[np.int8], slice, slice]
        The subsection of the section or the subsection and the slices used to
        extract it if return_slices is True.
    """
    # Vertical part
    depth_slice: slice = create_slice(
        section.shape[0], vertical_fraction, start[0]
    )

    # Horizontal part
    horizontal_fraction: float = subsection_length / section.shape[1]
    width_slice: slice = create_slice(
        section.shape[1], horizontal_fraction, start[1]
    )

    if return_slices:
        return section[depth_slice, width_slice], depth_slice, width_slice

    return section[depth_slice, width_slice]


def resize(sample: np.ndarray[np.int8],
           new_shape: tuple[int, int]
           ) -> np.ndarray[np.int8]:
    """
    Resizes a sample to a new shape using nearest neighbor interpolation.

    Parameters
    ----------
    sample : np.ndarray[np.int8]
        The sample to resize.
    new_shape : tuple[int, int]
        The new shape of the sample.

    Returns
    -------
    np.ndarray[np.int8]
        The resized sample.
    """
    src_rows, src_cols = sample.shape
    dst_rows, dst_cols = new_shape
    # Compute nearest neighbor indices
    row_indices: np.ndarray[int] = np.round(
        np.linspace(0, src_rows - 1, dst_rows)
    ).astype(int)
    col_indices: np.ndarray[int] = np.round(
        np.linspace(0, src_cols - 1, dst_cols)
    ).astype(int)

    # Use advanced indexing to select nearest neighbors
    resized_array: np.ndarray[np.int8] = sample[
        row_indices[:, None], col_indices
    ]
    return resized_array


def detransform(log_res: float | np.ndarray[float]
                ) -> float | np.ndarray[float]:
    """
    Maps a normalized log resistivity value to a resistivity value.

    The transformation is given by:
    :math:`\rho = 2 \times 10^{4 \times \rho_{log}}`

    Parameters
    ----------
    log_res : float or np.ndarray[float]
        The normalized log resistivity value(s).

    Returns
    -------
    float or np.ndarray[float]
        The resistivity value(s).
    """
    return 2 * 10 ** (4 * log_res)


def schlumberger_array(nbr_electrodes: int,
                       electrode_locations: np.ndarray[float],
                       data_type: str
                       ) -> list[dc.sources.Dipole]:
    """
    Creates a Schlumberger array.

    Parameters
    ----------
    nbr_electrodes : int
        The number of electrodes.
    electrode_locations : np.ndarray[float]
        The locations of the electrodes.
    data_type : str
        The type of data to collect.

    Returns
    -------
    list[dc.sources.Dipole]
        The list of sources.
    """
    source_list: list[dc.sources.Dipole] = []
    for a in range(1, nbr_electrodes // 2):
        locations_a: float = electrode_locations[:(-2 * a) - 1]
        locations_b: float = electrode_locations[(2 * a) + 1:]
        locations_m: float = electrode_locations[a:-a - 1]
        locations_n: float = electrode_locations[a + 1:-a]
        receivers_list_a: list[dc.receivers.Dipole] = [
            dc.receivers.Dipole(
                locations_m=loc_m, locations_n=loc_n, data_type=data_type
            )
            for loc_m, loc_n in zip(locations_m, locations_n)
        ]
        source_list_a: dc.sources.Dipole = [
            dc.sources.Dipole(
                receiver_list_a, location_a=loc_a, location_b=loc_b
            )
            for receiver_list_a, loc_a, loc_b in zip(
                receivers_list_a, locations_a, locations_b
            )
        ]
        source_list += source_list_a
    return source_list


def wenner_array(nbr_electrodes: int,
                 electrode_locations: np.ndarray[float],
                 data_type: str
                 ) -> list[dc.sources.Dipole]:
    """
    Creates a Wenner array.

    Parameters
    ----------
    nbr_electrodes : int
        The number of electrodes.
    electrode_locations : np.ndarray[float]
        The locations of the electrodes.
    data_type : str
        The type of data to collect.

    Returns
    -------
    list[dc.sources.Dipole]
        The list of sources.
    """
    source_list: list[dc.sources.Dipole] = []
    for a in range(1, (nbr_electrodes + 3) // 3 + 1):
        locations_a: float = electrode_locations[:-3 * a:]
        locations_b: float = electrode_locations[3 * a:]
        locations_m: float = electrode_locations[a:-2 * a:]
        locations_n: float = electrode_locations[2 * a:-a]
        receivers_list_a: list[dc.receivers.Dipole] = [
            dc.receivers.Dipole(
                locations_m=loc_m, locations_n=loc_n, data_type=data_type
            )
            for loc_m, loc_n in zip(locations_m, locations_n)
        ]
        source_list_a: list[dc.sources.Dipole] = [
            dc.sources.Dipole(
                receiver_list_a, location_a=loc_a, location_b=loc_b
            )
            for receiver_list_a, loc_a, loc_b in zip(
                receivers_list_a, locations_a, locations_b
            )
        ]

        source_list += source_list_a
    return source_list


def compute_active_columns(row: int,
                           is_even_row: bool,
                           total_cols: int,
                           offset: int,
                           ) -> tuple[int, int]:
    """
    Computes the active columns for a given row in a Wenner array.

    Parameters
    ----------
    row : int
        The row index.
    is_even_row : bool
        Indicates if the row is considered "even".
    total_cols : int
        The total number of columns.
    offset : int
        The offset to apply to the active columns.

    Returns
    -------
    tuple[int, int]
        The start and end columns.
    """
    if is_even_row:
        col_start: int = math.ceil(row * 1.5) - offset
        col_end: int = total_cols - math.ceil(row * 1.5) + offset
    else:
        col_start: int = math.ceil((row + 1) * 1.5) - 1 - offset
        col_end: int = total_cols - math.ceil((row + 1) * 1.5) + 1 + offset

    return col_start, col_end


def pseudosection_schlumberger(rhoa: list[float],
                               nbr_electrodes: int
                               ) -> np.ndarray[float]:
    """
    Creates a pseudo section from a forward model result using a Schlumberger
    array.

    Parameters
    ----------
    rhoa : list[float]
        The apparent resistivity values.
    nbr_electrodes : int
        The number of electrodes.

    Returns
    -------
    np.ndarray[float]
        The pseudo section.
    """
    # Pseudosection shape is all determined by the number of electrodes:
    # - Max number of columns is the number of electrodes minus 3.
    # - Number of rows is the number of electrodes divided by 2, minus 1.
    num_cols: int = nbr_electrodes - 3
    num_rows: int = nbr_electrodes // 2 - 1

    # Create the pseudo section, all non apparent resistivity values are set to
    # NaN.
    pseudo_section: np.ndarray[float] = np.empty(
        (num_rows, num_cols), dtype=float)
    pseudo_section.fill(np.nan)

    # Fill the pseudo section with the apparent resistivity values.
    value_index: int = 0
    for row_idx in range(num_rows):
        start_col: int = row_idx
        end_col: int = num_cols - row_idx
        num_values_this_row: int = end_col - start_col
        pseudo_section[row_idx, start_col:end_col] = rhoa[
            value_index: value_index + num_values_this_row
        ]
        value_index += num_values_this_row

    return pseudo_section


def pseudosection_wenner(rhoa: list[float],
                         nbr_electrodes: int
                         ) -> np.ndarray[float]:
    """
    Creates a pseudo section from a forward model result using a Wenner array.

    Parameters
    ----------
    rhoa : list[float]
        The apparent resistivity values.
    nbr_electrodes : int
        The number of electrodes.

    Returns
    -------
    np.ndarray[float]
        The pseudo section.
    """
    # Pseudosection shape is all determined by the number of electrodes:
    # - Max umber of columns is the number of electrodes minus 1.
    # - Number of rows is the number of electrodes minus 1, divided by 3.
    num_cols: int = nbr_electrodes - 3
    num_rows: int = (nbr_electrodes - 1) // 3

    # This flag indicates if the max number of columns is even.
    even_num_cols: bool = (num_cols % 2 == 0)
    if even_num_cols:
        # We want non even number of columns to be able to center the
        # triangle. If the first row has an even number of columns, we add
        # one to the number of columns.
        num_cols += 1

    offset: int = (nbr_electrodes - 1) % 2

    # Create the pseudo section, all non apparent resistivity values are set to
    # NaN.
    pseudo_section: np.ndarray[float] = np.empty(
        (num_rows, num_cols), dtype=float)
    pseudo_section.fill(np.nan)

    value_index: int = 0
    for row_idx in range(num_rows):
        # Determine if the current row is considered "even" based on num_cols
        # parity
        is_even_row = (row_idx % 2 == 0) \
            if even_num_cols else (row_idx % 2 == 1)

        col_start, col_end = compute_active_columns(
            row_idx, is_even_row, num_cols, offset)

        for col_idx in range(col_start, col_end):
            # For even rows, add a central cell which value interpolates the
            # two central values.
            if is_even_row and col_idx == (num_cols - 1) // 2:
                pseudo_section[row_idx, col_idx] = (
                    rhoa[value_index - 1] + rhoa[value_index]) / 2
            else:
                pseudo_section[row_idx, col_idx] = rhoa[value_index]
                value_index += 1

    return pseudo_section


def parse_input() -> Namespace:
    """
    Parse the input arguments.

    Returns
    -------
    Namespace
        The parsed arguments.
    """
    parser: ArgumentParser = ArgumentParser(
        description="Assign resistivity values to the mesh"
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to the data directory",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to the output directory",
    )
    parser.add_argument(
        "-vf",
        "--vertical_fraction",
        type=float,
        default=0.75,
        help="Fraction of the section to keep vertically."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print information about the forward models."
    )
    return parser.parse_args()


def assign_resistivity(resized_section: np.ndarray[np.int8],
                       ) -> tuple[np.ndarray[float],
                                  np.ndarray[float]]:
    """
    Assign resistivity values to the resized sections.

    Parameters
    ----------
    resized_section : np.ndarray[np.int8]
        The resized sections.
    section : np.ndarray[np.int8]
        The original section.

    Returns
    -------
    tuple[np.ndarray[float], np.ndarray[float]]
        The normalized log resistivity model and the resistivity model.
    """
    # Extract the rock classes from the original section
    rock_classes, inv = np.unique(resized_section, return_inverse=True)
    # Create a random normalized log resistivity value for each rock class
    norm_log_res_values: np.ndarray[float] = np.random.uniform(
        0, 1, size=len(rock_classes)
    )
    norm_log_resistivity_model = norm_log_res_values[inv].reshape(
        resized_section.shape
    )
    # Detransform the resistivity values
    resistivity_model = detransform(norm_log_resistivity_model[-1])
    return norm_log_resistivity_model, resistivity_model


def create_surveys(resistivity_model: np.ndarray[float],
                   NUM_ELECTRODES: int,
                   SCHEME_NAME: str,
                   LATERAL_PADDING: int,
                   ) -> dc.Survey:
    """
    Create the surveys for the simulations.

    Parameters
    ----------
    resistivity_model : np.ndarray[float]
        The resistivity model.
    NUM_ELECTRODES : int
        The number of electrodes.
    SCHEME_NAME : str
        The name of the scheme.
    LATERAL_PADDING : int
        The number of pixels to pad the lateral sides of the section.

    Returns
    -------
    dc.Survey
        The survey.
    """
    # Define the measurement data type
    data_type = "apparent_resistivity"

    # Define electrode locations
    electrode_locations_x = np.linspace(
        LATERAL_PADDING,
        resistivity_model.shape[1] - LATERAL_PADDING,
        NUM_ELECTRODES
    )
    electrode_locations_z = np.zeros_like(electrode_locations_x)
    electrode_locations = np.c_[
        electrode_locations_x, electrode_locations_z
    ]
    if SCHEME_NAME == "wa":
        array = wenner_array
    elif SCHEME_NAME == "slm":
        array = schlumberger_array
    else:
        raise NotImplementedError(
            f"Scheme name {SCHEME_NAME} not implemented."
        )
    source_list = array(NUM_ELECTRODES, electrode_locations, data_type)
    return dc.Survey(source_list)


def process_sample(section: np.ndarray[np.int8],
                   VERTICAL_FRACTION: float,
                   VERBOSE: bool,
                   sample: int = 0,
                   ) -> tuple[
                       int,
                       int,
                       str,
                       np.ndarray[float],
                       np.ndarray[float]
]:
    """
    Process a sample.

    Parameters
    ----------
    section : np.ndarray[np.int8]
        The section to process.
    VERTICAL_FRACTION : float
        The fraction of the section to keep vertically.
    VERBOSE : bool
        If True, print information about the forward models.
    sample : int, optional
        The sample index.

    Returns
    -------
    tuple[
        int,
        int,
        str,
        np.ndarray[float],
        np.ndarray[float]
    ]
        The number of electrodes, the subsection length, the scheme name, the
        pseudo section and the normalized log resistivity model.
    """
    # ----- 1. Define the parameters -----
    # Number of electrodes
    NUM_ELECTRODES: int = rd.randint(24, 96)
    # Scheme name
    SCHEME_NAME: str = rd.choice(["wa", "slm"])
    # Pixel between electrodes
    space_between_electrodes: int = 3
    # Lateral padding
    LATERAL_PADDING: int = 2
    LATERAL_PADDING *= space_between_electrodes
    # Resized length
    resized_length: int = (NUM_ELECTRODES - 1) + 2 * LATERAL_PADDING

    # ----- 2. Extract subsection from the section -----
    # The subsection length is randomly chosen between the number of
    # electrodes and the maximum length of the section.
    subsection_length: int = rd.randint(NUM_ELECTRODES, 200)
    sub_section = extract_subsection(
        section, subsection_length, VERTICAL_FRACTION
    )

    # ----- 3. Resize subsection -----
    resized_section: np.ndarray[np.int8] = resize(
        sub_section, (int(resized_length * VERTICAL_FRACTION), resized_length)
    )

    # ----- 4. Compute Log Resistivity -----
    norm_log_resistivity_model, resistivity_model = assign_resistivity(
        resized_section
    )

    # ----- 5. Flatten the resistivity model -----
    resistivity_model_flat: np.ndarray[float] = \
        np.flipud(resistivity_model).ravel()

    # ----- 6. Create a TensorMesh -----
    mesh: TensorMesh = TensorMesh(
        (
            [(1, resistivity_model.shape[1])],
            [(1, resistivity_model.shape[0])]
        ),
        origin="0N"
    )

    # ----- 7. Create the surveys -----
    survey: dc.Survey = create_surveys(
        resistivity_model, NUM_ELECTRODES, SCHEME_NAME, LATERAL_PADDING
    )

    # ----- 8. Create the topography -----
    # Create x coordinates
    x_topo = np.linspace(
        0, resistivity_model.shape[1], resistivity_model.shape[1]
    )
    # We want a flat topography
    z_topo = np.zeros_like(x_topo)
    # Create the 2D topography
    topo_2d: np.ndarray[float] = np.c_[x_topo, z_topo]

    # ----- 9. Link resistivities to the mesh -----
    # We activate all cells below the surface
    active_cells: np.ndarray = active_from_xyz(mesh, topo_2d)
    resistivity_map: maps.IdentityMap = maps.IdentityMap(mesh, mesh.n_cells)

    # ----- 10. Create the simulations -----
    survey.drape_electrodes_on_topography(
        mesh, active_cells, option="top", topography=topo_2d
    )
    _ = survey.set_geometric_factor()
    resistivity_simulation: dc.simulation_2d.Simulation2DNodal = \
        dc.simulation_2d.Simulation2DNodal(
            mesh,
            survey=survey,
            rhoMap=resistivity_map,
            verbose=VERBOSE,
            solver=Pardiso
        )

    # ----- 11. Compute the forward models -----
    forward_model = resistivity_simulation.dpred(
        resistivity_model_flat
    )

    # ----- 12. Recreate Pseudo Sections from Simulation Results -----
    if SCHEME_NAME == "wa":
        get_pseudosection = pseudosection_wenner
    elif SCHEME_NAME == "slm":
        get_pseudosection = pseudosection_schlumberger

    pseudosection = get_pseudosection(forward_model, NUM_ELECTRODES)

    return (
        NUM_ELECTRODES,
        subsection_length,
        SCHEME_NAME,
        pseudosection,
        norm_log_resistivity_model
    )


def main(NUM_SAMPLES: int,
         VERTICAL_FRACTION: float,
         DATA_PATH: Path,
         VERBOSE: bool
         ):
    """
    Main function to generate the pseudo sections.

    Parameters
    ----------
    NUM_SAMPLES : int
        The number of samples to generate.
    VERTICAL_FRACTION : float
        The fraction of the section to keep vertically.
    DATA_PATH : Path
        Path to the data directory.
    VERBOSE : bool
        If True, print information about the forward models.
    """
    # Open (or create) an LMDB environment.
    # Adjust map_size according to the expected total size of your data.
    env = lmdb.open('data.lmdb', map_size=10**9)  # Here, 1GB map size

    batch_size = 10  # Number of samples to store per transaction
    buffer = {}
    index = 0

    progress_bar = tqdm(
        total=NUM_SAMPLES,
        desc="Processing samples",
        unit="sample"
    )
    for file in DATA_PATH.glob("*.npz"):
        multi_array = np.load(file, mmap_mode="r")["arr_0"]
        for section in multi_array:
            sample = process_sample(
                    section,
                    VERTICAL_FRACTION,
                    VERBOSE,
                )
            data = pickle.dumps(sample)

            # Use a zero-padded key (as bytes) for ordering
            key = f"{index:08d}".encode('ascii')
            buffer[key] = data
            index += 1

            # Update tqdm progress bar
            progress_bar.update(1)

            # Flush the buffer to LMDB once we have enough samples
            if len(buffer) >= batch_size:
                with env.begin(write=True) as txn:
                    for k, v in buffer.items():
                        txn.put(k, v)
                buffer = {}  # Reset buffer after flushing

    # Write any remaining samples in the buffer
    if buffer:
        with env.begin(write=True) as txn:
            for k, v in buffer.items():
                txn.put(k, v)
    progress_bar.close()


def count_samples_in_file(filename: Path) -> int:
    """
    Count the number of samples in a file.

    Parameters
    ----------
    filename : Path
        Path to the file.

    Returns
    -------
    int
        The number of samples in the file.
    """
    return len(np.load(filename, mmap_mode="r")["arr_0"])


def count_samples(DATA_PATH: Path) -> int:
    """
    Count in parallel the number of samples in the dataset

    Parameters
    ----------
    DATA_PATH : Path
        Path to the data directory.

    Returns
    -------
    int
        The number of samples.
    """
    npz_files = list(DATA_PATH.glob("*.npz"))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        sample_counts = list(
            executor.map(count_samples_in_file, npz_files)
        )
    return sum(sample_counts)


if __name__ == "__main__":
    args: Namespace = parse_input()
    DATA_PATH: Path = args.data_path
    print(f"Data path set to {DATA_PATH}")

    OUTPUT_PATH: Path = args.output_path
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    print(f"Output path set to {OUTPUT_PATH}")

    VERTICAL_FRACTION: float = args.vertical_fraction
    print(f"Vertical fraction set to {VERTICAL_FRACTION}")

    with open(OUTPUT_PATH / "args.txt", "w") as f:
        f.write(f"Data path: {DATA_PATH}\n")
        f.write(f"Vertical fraction: {VERTICAL_FRACTION}\n")

    VERBOSE: bool = args.verbose

    NUM_SAMPLES: int = count_samples(DATA_PATH)

    pseudosections, timers, resistivity_models = main(
        NUM_SAMPLES,
        VERTICAL_FRACTION,
        DATA_PATH,
        VERBOSE
    )

    np.save(OUTPUT_PATH / "pseudosections.npy", np.array(pseudosections))
    np.save(OUTPUT_PATH / "timers.npy", timers)
    with open(OUTPUT_PATH / "resistivity_models.pkl", "wb") as f:
        pickle.dump(resistivity_models, f)
