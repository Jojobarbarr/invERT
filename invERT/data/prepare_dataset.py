
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
import random
from argparse import ArgumentParser, Namespace
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from typing import Optional, Tuple, List, Dict, Any

import traceback
import warnings
import os

# --- TQDM / Dummy TQDM ---
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    # Keep your DummyTqdm implementation
    class DummyTqdm:
        def __init__(self, iterable=None, *args, total=None, desc="Processing", unit="it", **kwargs):
            self.iterable = iterable
            self.total = total if total is not None else (len(iterable) if hasattr(iterable, '__len__') else 0)
            self.desc = desc
            self.unit = unit
            self.n = 0  # Progress counter
            # Determine update frequency (e.g., every 1% or every 1000 items, whichever is larger)
            self.mod = 1000
            self.start = perf_counter()
            self.last_print_n = 0
            self.last_print_time = self.start
            if self.total > 0:
                print(f"{self.desc}: Starting {self.total} {self.unit}.")

        def __iter__(self):
            if self.iterable is not None:
                for obj in self.iterable:
                    yield obj
                    self.update(1)
            else:
                return iter([])

        def update(self, n=1):
            self.n += n
            now = perf_counter()
            # Update display if enough items passed OR enough time passed (e.g., > 1 second)
            if (self.n - self.last_print_n >= self.mod or now - self.last_print_time > 3600.0) and self.total > 0:
                elapsed_time: float = now - self.start
                rate = self.n / elapsed_time if elapsed_time > 0 else 0
                remaining_items = self.total - self.n
                eta_seconds = remaining_items / rate if rate > 0 else 0
                progress_percent = (self.n / self.total) * 100

                print(
                    f"{self.desc}: {self.n}/{self.total} ({progress_percent:.1f}%) completed in "
                    f"{hhmmss(elapsed_time)} ({rate:.2f} {self.unit}/s). "
                    f"ETA: {hhmmss(eta_seconds)}"
                )
                self.last_print_n = self.n
                self.last_print_time = now

        def close(self):
            elapsed_time: float = perf_counter() - self.start
            rate = self.n / elapsed_time if elapsed_time > 0 else 0
            print(f"{self.desc}: Finished. Processed {self.n} {self.unit} in {hhmmss(elapsed_time)} ({rate:.2f} {self.unit}/s).")

    tqdm = DummyTqdm

# --- Suppress Warnings ---
# Suppress specific warnings - Keep this as it reduces noise
warnings.filterwarnings("ignore", category=DefaultSolverWarning)
warnings.filterwarnings("ignore", message="splu converted its input to CSC format")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Helper Functions (keep hhmmss, create_slice, extract_subsection, resize, detransform) ---
def hhmmss(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    h = int(h)
    m = int(m)
    s = int(s)
    return f"{h:02d}:{m:02d}:{s:02d}"

def create_slice(max_length: int,
                 fraction: float,
                 start: Optional[int] = None,
                 ) -> slice:
    if max_length == 0 or fraction == 0: return slice(0, 0) # Handle edge case
    slice_len = max(1, int(max_length * fraction)) # Ensure at least 1
    if start is None:
        # Ensure start index allows for full slice length
        max_start = max(0, max_length - slice_len)
        start = random.randint(0, max_start) if max_start > 0 else 0
    else:
        start = max(0, min(start, max_length - slice_len)) # Clamp start

    return slice(start, start + slice_len)


def extract_subsection(
        section: np.ndarray, # Removed specific type hint for flexibility
        subsection_length: int,
        vertical_fraction: float,
        start: Tuple[Optional[int], Optional[int]] = (None, None),
        return_slices: bool = False,
) -> np.ndarray | Tuple[np.ndarray, slice, slice]:

    if section.ndim != 2 or section.shape[0] == 0 or section.shape[1] == 0:
         # Handle empty or non-2D input
         empty_subsection = np.array([]).reshape(0, 0).astype(section.dtype)
         empty_slice = slice(0, 0)
         return (empty_subsection, empty_slice, empty_slice) if return_slices else empty_subsection

    depth_slice: slice = create_slice(
        section.shape[0], vertical_fraction, start[0]
    )

    horizontal_fraction: float = subsection_length / section.shape[1] if section.shape[1] > 0 else 0
    width_slice: slice = create_slice(
        section.shape[1], horizontal_fraction, start[1]
    )

    subsection = section[depth_slice, width_slice]

    if return_slices:
        return subsection, depth_slice, width_slice
    return subsection

def resize(sample: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    # Consider using cv2 or Pillow for potentially faster resizing if this is a bottleneck
    # import cv2
    # return cv2.resize(sample.astype(float), (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST).astype(sample.dtype)

    # Using NumPy implementation:
    src_rows, src_cols = sample.shape
    dst_rows, dst_cols = new_shape
    if src_rows == 0 or src_cols == 0: # Handle empty input
        return np.empty(new_shape, dtype=sample.dtype)

    row_indices: np.ndarray = np.round(
        np.linspace(0, src_rows - 1, dst_rows)
    ).astype(int)
    col_indices: np.ndarray = np.round(
        np.linspace(0, src_cols - 1, dst_cols)
    ).astype(int)
    # Clamp indices to avoid out-of-bounds errors if linspace goes slightly beyond due to float precision
    row_indices = np.clip(row_indices, 0, src_rows - 1)
    col_indices = np.clip(col_indices, 0, src_cols - 1)

    resized_array = sample[row_indices[:, None], col_indices]
    return resized_array

def detransform(log_res: np.ndarray) -> np.ndarray:
    # Ensure input is float for exponentiation
    return (2 * 10 ** (4 * log_res.astype(np.float64))).astype(log_res.dtype)


def schlumberger_array(nbr_electrodes: int,
                       electrode_locations: np.ndarray[np.float32],
                       data_type: str
                       ) -> list[dc.sources.Dipole]:
    """
    Creates a Schlumberger array.

    Parameters
    ----------
    nbr_electrodes : int
        The number of electrodes.
    electrode_locations : np.ndarray[np.float32]
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
                 electrode_locations: np.ndarray[np.float32],
                 data_type: str
                 ) -> list[dc.sources.Dipole]:
    """
    Creates a Wenner array.

    Parameters
    ----------
    nbr_electrodes : int
        The number of electrodes.
    electrode_locations : np.ndarray[np.float32]
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


def pseudosection_schlumberger(rhoa: list[np.float32],
                               nbr_electrodes: int
                               ) -> np.ndarray[np.float32]:
    """
    Creates a pseudo section from a forward model result using a Schlumberger
    array.

    Parameters
    ----------
    rhoa : list[np.float32]
        The apparent resistivity values.
    nbr_electrodes : int
        The number of electrodes.

    Returns
    -------
    np.ndarray[np.float32]
        The pseudo section.
    """
    # Pseudosection shape is all determined by the number of electrodes:
    # - Max number of columns is the number of electrodes minus 3.
    # - Number of rows is the number of electrodes divided by 2, minus 1.
    num_cols: int = nbr_electrodes - 3
    num_rows: int = nbr_electrodes // 2 - 1

    # Create the pseudo section, all non apparent resistivity values are set to
    # NaN.
    pseudo_section: np.ndarray[np.float32] = np.empty(
        (num_rows, num_cols), dtype=np.float32)
    pseudo_section.fill(0)

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


def pseudosection_wenner(rhoa: list[np.float32],
                         nbr_electrodes: int
                         ) -> np.ndarray[np.float32]:
    """
    Creates a pseudo section from a forward model result using a Wenner array.

    Parameters
    ----------
    rhoa : list[np.float32]
        The apparent resistivity values.
    nbr_electrodes : int
        The number of electrodes.

    Returns
    -------
    np.ndarray[np.float32]
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
    pseudo_section: np.ndarray[np.float32] = np.empty(
        (num_rows, num_cols), dtype=np.float32)
    pseudo_section.fill(0)

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


# --- Argument Parsing ---
def parse_input() -> Namespace:
    """Parses command-line arguments."""
    parser = ArgumentParser(description="Generate ERT Forward Models from Geological Sections")
    parser.add_argument("data_path", type=Path, help="Path to the input data directory (containing .npz sections)")
    parser.add_argument("output_path", type=Path, help="Path to the output directory for generated samples")
    parser.add_argument("-n", "--num_samples", type=int, default=-1,
                        help="Target number of samples to generate (-1 to process all available)")
    parser.add_argument("-vf", "--vertical_fraction", type=float, default=0.75,
                        help="Fraction of the vertical section to keep (0.0 to 1.0)")
    parser.add_argument("-w", "--workers", type=int, default=None, # Default to None for ProcessPoolExecutor default
                        help="Max number of worker processes (default: number of cores)")
    parser.add_argument("-cf", "--checkpoint_freq", type=int, default=1000,
                        help="Frequency (in samples) for writing checkpoints")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Enable verbose output during processing")

    args = parser.parse_args()

    # Validate paths
    if not args.data_path.is_dir():
        parser.error(f"Data path '{args.data_path}' not found or is not a directory.")
    args.output_path.mkdir(parents=True, exist_ok=True) # Ensure output exists

    # Validate fraction
    if not (0.0 < args.vertical_fraction <= 1.0):
         parser.error(f"Vertical fraction must be between 0.0 and 1.0 (exclusive of 0). Got: {args.vertical_fraction}")

    return args


# --- Core Processing Logic ---
def assign_resistivity(resized_section: np.ndarray[np.int8],
                       ) -> tuple[np.ndarray[np.float32],
                                  np.ndarray[np.float32]]:
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
    tuple[np.ndarray[np.float32], np.ndarray[np.float32]]
        The normalized log resistivity model and the resistivity model.
    """
    # Extract the rock classes from the original section
    rock_classes, inv = np.unique(resized_section, return_inverse=True)
    # Create a random normalized log resistivity value for each rock class
    norm_log_res_values: np.ndarray[np.float32] = np.random.uniform(
        0, 1, size=len(rock_classes)
    ).astype(np.float32)
    norm_log_resistivity_model = norm_log_res_values[inv].reshape(
        resized_section.shape
    )
    # Detransform the resistivity values
    resistivity_model = detransform(norm_log_resistivity_model)
    return norm_log_resistivity_model, resistivity_model


def create_surveys(resistivity_model: np.ndarray[np.float32],
                   NUM_ELECTRODES: int,
                   SCHEME_NAME: str,
                   LATERAL_PADDING: int,
                   ) -> dc.Survey:
    """
    Create the surveys for the simulations.

    Parameters
    ----------
    resistivity_model : np.ndarray[np.float32]
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
        NUM_ELECTRODES,
        dtype=np.float32,
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

def process_sample(section_data: Tuple[int, np.ndarray], # Pass index and data together
                   VERTICAL_FRACTION: float,
                   ) -> Optional[Dict[str, Any]]:
    worker_id = os.getpid() # Get worker pid for logging
    section_idx, section = section_data
    log_prefix = f"[Worker {worker_id}, Sample {section_idx}]"
    try:
        section_idx, section = section_data
        # ----- 1. Define the parameters -----
        NUM_ELECTRODES: int = random.randint(24, 96)
        SCHEME_NAME: str = random.choice(["wa", "slm"])
        space_between_electrodes: int = 5
        LATERAL_PADDING: int = 5 * space_between_electrodes
        resized_length: int = (NUM_ELECTRODES - 1) * space_between_electrodes + 2 * LATERAL_PADDING
        # ----- 2. Extract subsection -----
        # Ensure subsection_length is reasonable relative to section width
        max_subsection_len = section.shape[1] if section.ndim == 2 and section.shape[1] > NUM_ELECTRODES else NUM_ELECTRODES + 1
        min_subsection_len = NUM_ELECTRODES
        if min_subsection_len >= max_subsection_len:
                subsection_length = min_subsection_len
        else:
                subsection_length = random.randint(min_subsection_len, max_subsection_len)

        sub_section = extract_subsection(
            section, subsection_length, VERTICAL_FRACTION
        )
        # ----- 3. Resize subsection -----
        target_height = max(1, int(resized_length * VERTICAL_FRACTION)) # Ensure height >= 1
        resized_section: np.ndarray = resize(
            sub_section, (target_height, resized_length)
        )
        # ----- 4. Compute Log Resistivity -----
        norm_log_resistivity_model, resistivity_model = assign_resistivity(
            resized_section
        )
        # ----- 5. Flatten the resistivity model (only if mesh needs it) -----
        # SimPEG often works with flattened C-order arrays
        resistivity_model_flat = np.flipud(resistivity_model).ravel()
        # ----- 6. Create a TensorMesh -----
        # Make dimensions slightly more robust
        mesh_ny, mesh_nx = resistivity_model.shape
        mesh: TensorMesh = TensorMesh(
            (
                [(1.0, mesh_nx)], # Cell width of 1, nx cells
                [(1.0, mesh_ny)]  # Cell width of 1, ny cells
            ),
            origin="0N" # Origin at bottom-left
        )
        # ----- 7. Create the surveys -----
        survey: dc.Survey = create_surveys(
            resistivity_model, NUM_ELECTRODES, SCHEME_NAME, LATERAL_PADDING
        )
        # ----- 8. Create the topography -----
        x_topo = np.arange(mesh_nx + 1, dtype=np.float64) # Node locations
        z_topo = np.zeros_like(x_topo) # Flat topo at z=0 relative to mesh origin "N"
        topo_2d = np.c_[x_topo, z_topo]
        # ----- 9. Link resistivities to the mesh -----
        active_cells: np.ndarray = active_from_xyz(mesh, topo_2d) # Use Node reference
        # ----- 10. Create the simulations -----
        survey.drape_electrodes_on_topography(
            mesh, active_cells, option="top", topography=topo_2d
        )
        # Geometric factor calculation might throw warnings if electrodes are too close/far
        _ = survey.set_geometric_factor()

        # Use float64 for simulation stability
        simulation_rho_map = maps.IdentityMap(mesh)
        resistivity_simulation: dc.simulation_2d.Simulation2DNodal = \
            dc.simulation_2d.Simulation2DNodal(
                mesh,
                survey=survey,
                rhoMap=simulation_rho_map, # Use log-resistivity map for stability
                nky=8,
                verbose=False,
            )
        # ----- 11. Compute the forward models -----
        fields = resistivity_simulation.fields(resistivity_model_flat)
        forward_model = resistivity_simulation.dpred(resistivity_model_flat, f=fields).astype(np.float32)
        JtJ_diag = np.sum(np.square(resistivity_simulation.getJ(resistivity_model_flat, fields)), axis=0)
        JtJ_diag = np.flipud(JtJ_diag.reshape(resistivity_model.shape))

        # ----- 12. Recreate Pseudo Sections -----
        if SCHEME_NAME == "wa":
            get_pseudosection = pseudosection_wenner
        elif SCHEME_NAME == "slm":
            get_pseudosection = pseudosection_schlumberger

        if np.min(forward_model) < 0:
            print(f"{log_prefix} Warning: Negative forward model values for sample {section_idx}. Skipping.")
            return None
        
        pseudosection = get_pseudosection(forward_model, NUM_ELECTRODES).astype(np.float32)
        # ----- 13. Package Results -----
        # Convert to tensors just before returning
        sample_out = {
            'num_electrode': np.int32(NUM_ELECTRODES),
            'subsection_length': np.int32(subsection_length),
            'array_type': np.array([1, 0] if SCHEME_NAME == "wa" else [0, 1], dtype=np.int32), # One-hot encode
            'pseudosection': pseudosection,
            'norm_log_resistivity_model': norm_log_resistivity_model,
            'JtJ_diag': JtJ_diag,
        }
        return sample_out
    
    except Exception as e:
        print(f"{log_prefix} Exception occurred in processing sample {section_data[0]}:")
        traceback.print_exc()  # This prints the full traceback to stdout
        raise  # Re-raise the exception to let the process crash

def write_checkpoint(output_path: Path, count: int, last_file: str):
    """Writes a simple checkpoint file."""
    try:
        ckpt_file = output_path / "ckpt.txt"
        with open(ckpt_file, "w") as f:
            f.write(f"{count}\n")
            f.write(f"{last_file}\n")
    except Exception as e:
        print(f"Warning: Failed to write checkpoint file {ckpt_file}: {e}")

def main_parallel(NUM_SAMPLES_TO_GENERATE: int,
                  VERTICAL_FRACTION: float,
                  DATA_PATH: Path,
                  OUTPUT_PATH: Path, # This is now the DIRECTORY for .npz files
                  MAX_WORKERS: Optional[int],
                  CHECKPOINT_FREQUENCY: int = 1000):
    print(f"Starting parallel processing to generate .npz files.")
    print(f"  Output Directory: {OUTPUT_PATH}")
    print(f"  Max Workers: {MAX_WORKERS if MAX_WORKERS else 'Default'}")
    print(f"  Checkpoint Frequency: Approx every {CHECKPOINT_FREQUENCY} samples")

    # --- Setup Output Directory ---
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {OUTPUT_PATH}")
    sample_path = OUTPUT_PATH / "samples"
    sample_path.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    last_ckpt_count = 0
    # ckpt_frequency = BATCH_SIZE * 10 # Checkpoint frequency now an argument or fixed
    last_processed_file = "N/A"

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        total_tasks_submitted = 0

        files = sorted(list(DATA_PATH.glob("*.npz"))) # Sort for deterministic order
        print(f"Found {len(files)} input '.npz' files.")

        # --- Submission Phase ---
        global_section_idx = 0
        stop_submission = False
        for file_idx, file in enumerate(files):
            if stop_submission:
                break
            last_processed_file = file.name
            try:
                # Use mmap_mode for potentially large files, handle potential load errors
                multi_array = np.load(file, mmap_mode="r")["arr_0"]
            except Exception as e:
                print(f"Error loading file {file.name}: {e}. Skipping.")
                continue

            num_sections_in_file = multi_array.shape[0]
            print(f"Submitting tasks from {file.name} ({num_sections_in_file} sections)...")

            for local_idx in range(num_sections_in_file):
                # Package data for submission
                section_data_tuple = (global_section_idx, multi_array[local_idx])

                future = executor.submit(
                    process_sample,
                    section_data_tuple,
                    VERTICAL_FRACTION,
                )
                futures.append(future)
                global_section_idx += 1
                total_tasks_submitted += 1

                # Stop submitting if we reach the target number of samples
                if total_tasks_submitted >= NUM_SAMPLES_TO_GENERATE:
                    print(f"Reached target number of samples ({NUM_SAMPLES_TO_GENERATE}), stopping submission.")
                    stop_submission = True
                    break # Stop processing this file

        # --- Processing Phase ---
        print(f"Submitted {total_tasks_submitted} tasks. Processing results and saving .npz files...")
        progress_bar = tqdm(total=len(futures), desc="Saving samples", unit="sample")

        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                sample_result = future.result()
            except Exception as e:
                print(f"Error processing future {idx}: {e}. Skipping.")
                traceback.print_exc()  # This prints the full traceback to stdout
                # Optionally log more details about which input sample failed
                raise # Re-raise the exception to let the process crash

            if sample_result is None:
                print(f"Skipping sample {idx} (worker returned None).")
                continue # Skip samples that failed in worker

            # --- Save as .npz file ---
            # Generate unique filename using the processed count
            output_filename = sample_path / f"sample_{processed_count:08d}.npz"
            np.savez_compressed(output_filename, **sample_result)

            processed_count += 1
            progress_bar.update(1)

            # --- Checkpointing (Independent of saving) ---
            if processed_count >= last_ckpt_count + CHECKPOINT_FREQUENCY:
                print(f"Writing checkpoint at {processed_count} samples.")
                # Make sure write_checkpoint handles potential errors
                try:
                    write_checkpoint(OUTPUT_PATH, processed_count, last_processed_file)
                    last_ckpt_count = processed_count
                except Exception as e:
                    print(f"Error writing checkpoint: {e}")

        # --- Final Actions ---
        # Write final checkpoint after loop finishes
        print(f"Writing final checkpoint at {processed_count} samples.")
        try:
            write_checkpoint(OUTPUT_PATH, processed_count, last_processed_file)
        except Exception as e:
             print(f"Error writing final checkpoint: {e}")

    progress_bar.close()
    print(f"Processing complete. Total samples written to {OUTPUT_PATH}: {processed_count}")

def count_samples_in_file(filename: Path) -> int:
    """Count samples using shape."""
    with np.load(filename, mmap_mode="r") as data:
        if "arr_0" in data:
            return data["arr_0"].shape[0]
        else:
            print(f"Warning: 'arr_0' not found in {filename}")
            return 0

def count_samples(DATA_PATH: Path) -> int:
    """Count total samples using ThreadPoolExecutor for I/O bound task."""
    npz_files = list(DATA_PATH.glob("*.npz"))
    if not npz_files:
        print("No .npz files found.")
        return 0

    total_samples = 0
    # Use ThreadPool for I/O bound task (reading file headers)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(count_samples_in_file, f): f for f in npz_files}
        results_iterator = concurrent.futures.as_completed(future_to_file)
        results_iterator = tqdm(results_iterator, total=len(npz_files), desc="Counting samples", unit="file")

        for future in results_iterator:
            count = future.result()
            total_samples += count
    return total_samples


if __name__ == "__main__":
    args: Namespace = parse_input()
    DATA_PATH: Path = args.data_path.resolve() # Use absolute path
    print(f"Data path set to {DATA_PATH}")

    OUTPUT_PATH: Path = args.output_path.resolve()
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    print(f"Output path set to {OUTPUT_PATH}")

    VERTICAL_FRACTION: float = args.vertical_fraction
    print(f"Vertical fraction set to {VERTICAL_FRACTION}")

    MAX_WORKERS = args.workers

    # Save config args
    with open(OUTPUT_PATH / "data_args.txt", "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    print(f"Counting samples in {DATA_PATH}...")
    # We don't necessarily need to count all samples beforehand if we generate a fixed number
    NUM_SAMPLES_AVAILABLE: int = count_samples(DATA_PATH)
    print(f"Total samples available in source files: {NUM_SAMPLES_AVAILABLE}")

    # Decide how many samples to generate (e.g., from args or fixed number)
    NUM_SAMPLES_TO_GENERATE = NUM_SAMPLES_AVAILABLE # Example: Generate 10k samples
    print(f"Attempting to generate {NUM_SAMPLES_TO_GENERATE} samples.")

    main_parallel(
        NUM_SAMPLES_TO_GENERATE, # Target number
        VERTICAL_FRACTION,
        DATA_PATH,
        OUTPUT_PATH,
        MAX_WORKERS,
    )