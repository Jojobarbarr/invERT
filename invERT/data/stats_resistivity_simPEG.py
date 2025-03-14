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
from time import perf_counter
from tqdm import tqdm
import random as rd
from argparse import ArgumentParser, Namespace
import concurrent.futures
from functools import partial
import pickle
from os import cpu_count

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


def sample_section(DATASET_PATH: Path,
                   npz_keys: list[int],
                   already_selected: set[tuple[int, int]],
                   ) -> np.ndarray[np.int8]:
    """
    Samples a section from the dataset.

    Parameters
    ----------
    DATASET_PATH : Path
        Path to the dataset.
    npz_keys : list[int]
        List of keys for the npz files.
    already_selected : set[tuple[int, int]]
        Set of already selected sections.
    Returns
    -------
    np.ndarray[np.int8]
        The section.
    """
    while True:
        # Chooses the file from which to sample.
        npz_idx: int = rd.choice(npz_keys)
        npz_path: Path = DATASET_PATH / f"{npz_idx}.npz"
        # Loads the file.
        multi_array: np.ndarray[np.int8] = \
            np.load(npz_path, mmap_mode="r")["arr_0"]

        # Chooses the section from the file.
        section_idx: int = rd.choice(range(multi_array.shape[0]))
        section: np.ndarray[np.int8] = multi_array[section_idx]

        # Checks if the section has not already been selected and returns it.
        if (npz_idx, section_idx) not in already_selected:
            already_selected.add(
                (npz_idx, section_idx)
            )
            return section


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


def transform(res: float | np.ndarray[float]
              ) -> float | np.ndarray[float]:
    """"
    Maps a resistivity value to a normalized log resistivity value.

    The transformation is given by:
    :math:`\rho_{log} = \\log_{10}(\rho / 2) / 4`

    Parameters
    ----------
    res : float or np.ndarray[float]
        The resistivity value(s).

    Returns
    -------
    float or np.ndarray[float]
        The normalized log resistivity value(s).
    """
    return np.log10(res / 2) / 4


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
        "-n",
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "-e",
        "--num_electrodes",
        type=int,
        default=48,
        help="Number of electrodes to use."
    )
    parser.add_argument(
        "-s",
        "--space_between_electrodes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Spacing between electrodes."
    )
    parser.add_argument(
        "-sn",
        "--scheme_name",
        type=str,
        default="wa",
        help="Name of the scheme."
    )
    parser.add_argument(
        "-vf",
        "--vertical_fraction",
        type=float,
        default=0.5,
        help="Fraction of the section to keep vertically."
    )
    parser.add_argument(
        "-np",
        "--non_parallel",
        action="store_true",
        help="Do not use parallel processing."
    )
    parser.add_argument(
        "-w",
        "--num_max_workers",
        type=int,
        default=cpu_count(),
        help="Number of workers to use."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print information about the forward models."
    )
    return parser.parse_args()


def resize_sections(section: np.ndarray[np.int8],
                    resized_lengths: np.ndarray[int]
                    ) -> list[np.ndarray[np.int8]]:
    """
    Resize the sections to the given lengths.

    Parameters
    ----------
    sections : np.ndarray[np.int8]
        The sections to resize.
    resized_lengths : np.ndarray[int]
        The lengths to resize the sections to.

    Returns
    -------
    list[np.ndarray[np.int8]]
        The resized sections.
    """
    return [
        resize(section, (section.shape[0], resize_length))
        for resize_length in resized_lengths
    ]


def assign_resistivity(resized_sections: list[np.ndarray[np.int8]],
                       section: np.ndarray[np.int8]
                       ) -> tuple[list[np.ndarray[float]],
                                  list[np.ndarray[float]]]:
    """
    Assign resistivity values to the resized sections.

    Parameters
    ----------
    resized_sections : list[np.ndarray[np.int8]]
        The resized sections.
    section : np.ndarray[np.int8]
        The original section.

    Returns
    -------
    tuple[list[np.ndarray[float]], list[np.ndarray[float]]]
        The normalized log resistivity models and the resistivity models.
    """
    # Extract the rock classes from the original section
    rock_classes = np.unique(section)
    # Create a random normalized log resistivity value for each rock class
    norm_log_res_values: np.ndarray[float] = np.random.uniform(
        0, 1, size=len(rock_classes)
    )
    # Assign the random log resistivity value to each pixel according to
    # its rock class
    norm_log_resistivity_models: list[np.ndarray[float]] = []
    resistivity_models: list[np.ndarray[float]] = []
    for resized_section in resized_sections:
        # Get the mapping between rock classes and pixels
        _, inv = np.unique(resized_section, return_inverse=True)
        # Create a normalized log resistivity model
        norm_log_resistivity_models.append(
            norm_log_res_values[inv].reshape(resized_section.shape)
        )
        # Detransform the resistivity values
        resistivity_models.append(
            detransform(norm_log_resistivity_models[-1])
        )
    return norm_log_resistivity_models, resistivity_models


def flatten_models(resistivity_models: list[np.ndarray[float]]
                   ) -> list[np.ndarray[float]]:
    """
    Flatten the resistivity models.

    Parameters
    ----------
    resistivity_models : list[np.ndarray[float]]
        The resistivity models.

    Returns
    -------
    list[np.ndarray[float]]
        The flattened resistivity models.
    """
    return [
        np.flipud(resistivity_model).ravel()
        for resistivity_model in resistivity_models
    ]


def create_meshes(resistivity_models: list[np.ndarray[float]]
                  ) -> list[TensorMesh]:
    """
    Create the meshes for the simulations.

    Parameters
    ----------
    resistivity_models : list[np.ndarray[float]]
        The resistivity models.

    Returns
    -------
    list[TensorMesh]
        The meshes.
    """
    return [
        TensorMesh(
            (
                [(1, resistivity_model.shape[1])],
                [(1, resistivity_model.shape[0])]
            ),
            origin="0N"
        )
        for resistivity_model in resistivity_models
    ]


def create_surveys(resistivity_models: list[np.ndarray[float]],
                   NUM_ELECTRODES: int,
                   SCHEME_NAME: str,
                   ) -> list[dc.Survey]:
    """
    Create the surveys for the simulations.

    Parameters
    ----------
    resistivity_models : list[np.ndarray[float]]
        The resistivity models.
    NUM_ELECTRODES : int
        The number of electrodes.
    SCHEME_NAME : str
        The name of the scheme.

    Returns
    -------
    list[dc.Survey]
        The surveys.
    """
    # Define the measurement data type
    data_type = "apparent_resistivity"

    surveys: list[dc.Survey] = []
    for resistivity_model in resistivity_models:
        # Define electrode locations
        electrode_locations_x = np.linspace(
            0, resistivity_model.shape[1], NUM_ELECTRODES
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
        # Define survey
        surveys.append(dc.Survey(source_list))
    return surveys


def create_topo2ds(resistivity_models: list[np.ndarray[float]]
                   ) -> list[np.ndarray[float]]:
    """
    Create the topography for the simulations.

    Parameters
    ----------
    resistivity_models : list[np.ndarray[float]]
        The resistivity models.

    Returns
    -------
    list[np.ndarray[float]]
        The topographies.
    """
    topo_2ds: list[np.ndarray[float]] = []
    for resistivity_model in resistivity_models:
        # Create x coordinates
        x_topo = np.linspace(
            0, resistivity_model.shape[1], resistivity_model.shape[1]
        )
        # We want a flat topography
        z_topo = np.zeros_like(x_topo)
        # Create the 2D topography
        topo_2ds.append(np.c_[x_topo, z_topo])
    return topo_2ds


def create_simulations(surveys: list[dc.Survey],
                       meshes: list[TensorMesh],
                       actived_cells_list: list[np.ndarray],
                       topo_2ds: list[np.ndarray[float]],
                       resistivity_maps: list[maps.IdentityMap],
                       verbose: bool,
                       ) -> list[dc.simulation_2d.Simulation2DNodal]:
    """
    Create the simulations for the resistivity models.

    Parameters
    ----------
    surveys : list[dc.Survey]
        The surveys.
    meshes : list[TensorMesh]
        The meshes.
    actived_cells_list : list[np.ndarray]
        The active cells.
    topo_2ds : list[np.ndarray[float]]
        The topographies.
    resistivity_maps : list[maps.IdentityMap]
        The resistivity maps.
    verbose : bool
        If True, print information about the forward models.

    Returns
    -------
    list[dc.simulation_2d.Simulation2DNodal]
        The simulations.
    """
    for survey, mesh, active_cells, topo_2d in zip(
        surveys, meshes, actived_cells_list, topo_2ds
    ):
        survey.drape_electrodes_on_topography(
            mesh, active_cells, option="top", topography=topo_2d
        )
        _ = survey.set_geometric_factor()

    resistivity_simulations: list[dc.simulation_2d.Simulation2DNodal] = [
        dc.simulation_2d.Simulation2DNodal(
            mesh, survey=survey, rhoMap=resistivity_map, verbose=verbose
        )
        for mesh, survey, resistivity_map in zip(
            meshes, surveys, resistivity_maps
        )
    ]
    return resistivity_simulations


def compute_forward_models(
        resistivity_simulations: list[dc.simulation_2d.Simulation2DNodal],
        resistivity_models_flat: list[np.ndarray[float]],
) -> tuple[list[np.ndarray[float]], np.ndarray[float]]:
    """
    Compute the forward models for the resistivity simulations.

    Parameters
    ----------
    resistivity_simulations : list[dc.simulation_2d.Simulation2DNodal]
        The resistivity simulations.
    resistivity_models_flat : list[np.ndarray[float]]
        The flattened resistivity models.

    Returns
    -------
    tuple[list[np.ndarray[float]], np.ndarray[float]]
        The forward models and the time taken to compute them.
    """
    forward_models: list[np.ndarray[float]] = []
    timer_arr: np.ndarray[float] = np.zeros(
        len(resistivity_simulations), dtype=float
    )
    for idx, (resistivity_simulation, resistivity_model_flat) in enumerate(
        zip(
            resistivity_simulations, resistivity_models_flat
        )
    ):
        start_time = perf_counter()
        forward_model = resistivity_simulation.dpred(
            resistivity_model_flat
        )
        stop_time = perf_counter()
        forward_models.append(forward_model)
        timer_arr[idx] = stop_time - start_time
    return forward_models, timer_arr


def process_sample(DATA_PATH: Path,
                   npz_keys: list[int],
                   already_selected: set[tuple[int, int]],
                   NUM_ELECTRODES: int,
                   SCHEME_NAME: str,
                   VERTICAL_FRACTION: float,
                   AIR_RESISTIVITY: float,
                   resized_lengths: np.ndarray[int],
                   verbose: bool,
                   sample: int,
                   ) -> tuple[
                       np.ndarray[float],
                       np.ndarray[float],
                       np.ndarray[float]
]:
    """
    Process a sample.

    Parameters
    ----------
    DATA_PATH : Path
        Path to the data directory.
    npz_keys : list[int]
        List of keys for the npz files.
    already_selected : set[tuple[int, int]]
        Set of already selected sections.
    NUM_ELECTRODES : int
        The number of electrodes.
    SCHEME_NAME : str
        The name of the scheme.
    VERTICAL_FRACTION : float
        The fraction of the section to keep vertically.
    AIR_RESISTIVITY : float
        The resistivity of the air.
    resized_lengths : np.ndarray[int]
        The lengths to resize the sections to.
    verbose : bool
        If True, print information about the forward models.
    sample : int
        The sample index. Th

    Returns
    -------
    tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]
        The pseudo section, the timer and the resistivity model.
    """
    # ----- 1. Select a unique section -----
    section: np.ndarray[np.int8] = sample_section(
        DATA_PATH,
        npz_keys,
        already_selected
    )

    # ----- 2. Extract subsection from the section -----

    # The subsection length is randomly chosen between the number of
    # electrodes and the maximum length of the section.
    subsection_length: int = rd.randint(NUM_ELECTRODES, 200)
    sub_section = extract_subsection(
        section, subsection_length, VERTICAL_FRACTION
    )
    # ----- 3. Resize subsection for Each Spacing Value -----
    resized_sections: list[np.ndarray[np.int8]] = resize_sections(
        sub_section, resized_lengths
    )

    # ----- 4. Compute Log Resistivity for Each Resized Section -----
    norm_log_resistivity_models, resistivity_models = assign_resistivity(
        resized_sections, section
    )

    # ----- 6. Flatten the resistivity models -----
    resistivity_models_flat: list[np.ndarray[float]] = flatten_models(
        resistivity_models
    )

    # ----- 7. Create a TensorMesh -----
    meshes: list[TensorMesh] = create_meshes(
        resistivity_models
    )

    # ----- 8. Create the surveys -----
    surveys: list[dc.Survey] = create_surveys(
        resistivity_models, NUM_ELECTRODES, SCHEME_NAME
    )

    # ----- 9. Create the topography -----
    topo_2ds: list[np.ndarray[float]] = create_topo2ds(
        resistivity_models
    )

    # ----- 10. Link resistivities to the mesh -----
    # We activate all cells below the surface
    actived_cells_list: list[np.ndarray] = [
        active_from_xyz(mesh, topo_2d)
        for mesh, topo_2d in zip(meshes, topo_2ds)
    ]
    # Compute the number of active cells
    resistivity_maps: list[maps.IdentityMap] = [
        maps.InjectActiveCells(mesh, active_cells, AIR_RESISTIVITY)
        for mesh, active_cells in zip(meshes, actived_cells_list)
    ]

    # ----- 11. Create the simulations -----
    resistivity_simulations: list[list[dc.simulation_2d.Simulation2DNodal]] = \
        create_simulations(
            surveys,
            meshes,
            actived_cells_list,
            topo_2ds,
            resistivity_maps,
            verbose
    )

    # ----- 12. Compute the forward models -----
    forward_models, timers = compute_forward_models(
        resistivity_simulations, resistivity_models_flat
    )

    # ----- 13. Recreate Pseudo Sections from Simulation Results -----
    if SCHEME_NAME == "wa":
        pseudosection = pseudosection_wenner
    elif SCHEME_NAME == "slm":
        pseudosection = pseudosection_schlumberger

    pseudosections = [
        pseudosection(forward_model, NUM_ELECTRODES)
        for forward_model in forward_models
    ]

    return pseudosections, timers, resistivity_models


def main(NUM_SAMPLES: int,
         NUM_ELECTRODES: int,
         SCHEME_NAME: str,
         VERTICAL_FRACTION: float,
         AIR_RESISTIVITY: float,
         DATA_PATH: Path,
         resized_lengths: np.ndarray[int],
         VERBOSE: bool
         ) -> tuple[
             list[np.ndarray[float]],
             list[np.ndarray[float]],
             list[np.ndarray[float]]
]:
    """
    Main function to generate the pseudo sections.

    Parameters
    ----------
    NUM_SAMPLES : int
        The number of samples to generate.
    NUM_ELECTRODES : int
        The number of electrodes.
    SCHEME_NAME : str
        The name of the scheme.
    VERTICAL_FRACTION : float
        The fraction of the section to keep vertically.
    AIR_RESISTIVITY : float
        The resistivity of the air.
    DATA_PATH : Path
        Path to the data directory.
    resized_lengths : np.ndarray[int]
        The lengths to resize the sections to.
    VERBOSE : bool
        If True, print information about the forward models.

    Returns
    -------
    tuple[
        list[np.ndarray[float]],
        list[np.ndarray[float]],
        list[np.ndarray[float]]
    ]
        The pseudo sections, the timers and the resistivity models.
    """
    # Build set for available npz files.
    npz_keys: list[int] = [int(npz.stem) for npz in DATA_PATH.glob("*.npz")]

    # Keep track of already selected sections.
    already_selected: set[tuple[int, int]] = set()

    # This will hold the pseudo sections for all samples.
    pseudosections_list: list[np.ndarray[float]] = []
    # This will hold the time taken to compute the forward models for each
    # sample.
    timers_list: list[np.ndarray[float]] = []
    resistivity_models_list: list[np.ndarray[float]] = []

    for sample in tqdm(range(NUM_SAMPLES),
                       desc="Processing samples",
                       unit="sample"):
        pseudosections, timers, resistivity_models = process_sample(
            DATA_PATH,
            npz_keys,
            already_selected,
            NUM_ELECTRODES,
            SCHEME_NAME,
            VERTICAL_FRACTION,
            AIR_RESISTIVITY,
            resized_lengths,
            VERBOSE,
            sample,
        )
        pseudosections_list.append(pseudosections)
        timers_list.append(timers)
        resistivity_models_list.append(resistivity_models)
    return pseudosections_list, timers_list, resistivity_models_list


def main_parallel(NUM_SAMPLES: int,
                  NUM_ELECTRODES: int,
                  SCHEME_NAME: str,
                  VERTICAL_FRACTION: float,
                  AIR_RESISTIVITY: float,
                  DATA_PATH: Path,
                  resized_lengths: np.ndarray,
                  num_max_workers: int,
                  VERBOSE: bool,
                  
                  ) -> tuple[
                      list[np.ndarray[float]],
                      list[np.ndarray[float]],
                      list[np.ndarray[float]]
]:
    """
    Main function to generate the pseudo sections in parallel.

    Parameters
    ----------
    NUM_SAMPLES : int
        The number of samples to generate.
    NUM_ELECTRODES : int
        The number of electrodes.
    SCHEME_NAME : str
        The name of the scheme.
    VERTICAL_FRACTION : float
        The fraction of the section to keep vertically.
    AIR_RESISTIVITY : float
        The resistivity of the air.
    DATA_PATH : Path
        Path to the data directory.
    resized_lengths : np.ndarray[int]
        The lengths to resize the sections to.
    num_max_workers : int
        The number of workers to use.
    VERBOSE : bool
        If True, print information about the forward models.

    Returns
    -------
    tuple[
        list[np.ndarray[float]],
        list[np.ndarray[float]],
        list[np.ndarray[float]]
    ]
        The pseudo sections, the timers and the resistivity models.
    """
    # Build set for available npz files.
    npz_keys = [int(npz.stem) for npz in DATA_PATH.glob("*.npz")]

    # If samples are independent, each process can get its own copy of the set.
    already_selected = set()

    # Use partial to fix all arguments except the sample index.
    process_sample_partial = partial(
        process_sample,
        DATA_PATH,
        npz_keys,
        already_selected,
        NUM_ELECTRODES,
        SCHEME_NAME,
        VERTICAL_FRACTION,
        AIR_RESISTIVITY,
        resized_lengths,
        VERBOSE
    )

    pseudosections_list = []
    timers_list = []
    resistivity_models_list = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_max_workers
    ) as executor:
        # Using executor.map preserves the order of results.
        results = list(
            tqdm(
                executor.map(process_sample_partial, range(NUM_SAMPLES)),
                total=NUM_SAMPLES,
                desc="Processing samples",
                unit="sample"
            )
        )

    # Each result is expected to be a tuple (pseudosections, timers)
    for pseudosections, timers, resistivity_models in results:
        pseudosections_list.append(pseudosections)
        timers_list.append(timers)
        resistivity_models_list.append(resistivity_models)

    return pseudosections_list, timers_list, resistivity_models_list


if __name__ == "__main__":
    args: Namespace = parse_input()
    DATA_PATH: Path = args.data_path
    print(f"Data path set to {DATA_PATH}")

    OUTPUT_PATH: Path = args.output_path
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    print(f"Output path set to {OUTPUT_PATH}")

    NUM_SAMPLES: int = args.num_samples
    print(f"Number of samples set to {NUM_SAMPLES}")

    NUM_ELECTRODES: int = args.num_electrodes
    print(f"Number of electrodes set to {NUM_ELECTRODES}")

    SPACE_BETWEEN_ELECTRODES_LIST: np.ndarray[int] = np.array(
        args.space_between_electrodes, dtype=int
    )
    print(f"Spaces between electrodes: {SPACE_BETWEEN_ELECTRODES_LIST}")
    print(f"({len(SPACE_BETWEEN_ELECTRODES_LIST)} forward passes per sample)")

    SCHEME_NAME: str = args.scheme_name
    print(f"Scheme name set to {SCHEME_NAME}")

    VERTICAL_FRACTION: float = args.vertical_fraction
    print(f"Vertical fraction set to {VERTICAL_FRACTION}")

    AIR_RESISTIVITY: float = 1e8

    with open(DATA_PATH / "args.txt", "w") as f:
        f.write(f"Data path: {DATA_PATH}\n")
        f.write(f"Number of samples: {NUM_SAMPLES}\n")
        f.write(f"Number of electrodes: {NUM_ELECTRODES}\n")
        f.write(
            f"Piwels between electrodes: {SPACE_BETWEEN_ELECTRODES_LIST}\n"
        )
        f.write(f"Scheme name: {SCHEME_NAME}\n")
        f.write(f"Vertical fraction: {VERTICAL_FRACTION}\n")

    PARALLEL: bool = not args.non_parallel

    VERBOSE: bool = args.verbose
    if PARALLEL and VERBOSE:
        warnings.warn(
            "\nVerbose mode is not advised when using parallel processing"
            " as it may lead to a lot of output difficult to interpret.\n"
            "Consider using -np flag to disable parallel processing if you "
            "want to see the output of the forward models, or disable the "
            "verbose mode.\n"
        )

    # Calculate total pixels to keep for each spacing.
    resized_lengths: np.ndarray[np.int32] = \
        (NUM_ELECTRODES - 1) * SPACE_BETWEEN_ELECTRODES_LIST

    if PARALLEL:
        MAX_NUM_WORKERS: int = args.num_max_workers
        print(f"Number of workers set to {MAX_NUM_WORKERS}")
        pseudosections, timers, resistivity_models = main_parallel(
            NUM_SAMPLES,
            NUM_ELECTRODES,
            SCHEME_NAME,
            VERTICAL_FRACTION,
            AIR_RESISTIVITY,
            DATA_PATH,
            resized_lengths,
            MAX_NUM_WORKERS,
            VERBOSE
        )
    else:
        pseudosections, timers, resistivity_models = main(
            NUM_SAMPLES,
            NUM_ELECTRODES,
            SCHEME_NAME,
            VERTICAL_FRACTION,
            AIR_RESISTIVITY,
            DATA_PATH,
            resized_lengths,
            VERBOSE
        )

    np.save(OUTPUT_PATH / "pseudosections.npy", np.array(pseudosections))
    np.save(OUTPUT_PATH / "timers.npy", timers)
    with open(OUTPUT_PATH / "resistivity_models.pkl", "wb") as f:
        pickle.dump(resistivity_models, f)
