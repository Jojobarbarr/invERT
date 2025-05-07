from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps
from simpeg.utils.solver_utils import DefaultSolverWarning
from discretize import TensorMesh
from discretize.utils import active_from_xyz

from tqdm import tqdm
import numpy as np
from pathlib import Path
import math
from time import perf_counter
import random as rd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes._axes import Axes
from matplotlib.colors import LogNorm, Normalize

import warnings
warnings.filterwarnings("ignore", category=DefaultSolverWarning)
warnings.filterwarnings("ignore", message="splu converted its input to CSC format")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def detransform(log_res: float | np.ndarray[float]) -> float | np.ndarray[float]:
    return 2 * 10 ** (4 * log_res)
def transform(res: float | np.ndarray[float]) -> float | np.ndarray[float]:
    return np.log10(res / 2) / 4


def create_colorbar(sample: np.ndarray[np.int8]) -> tuple[dict[np.int8, tuple[float, float, float, float]], mcolors.BoundaryNorm, mcolors.ListedColormap]:
    classes: np.ndarray[np.int8] = np.unique(sample)
    cmap: mcolors.ListedColormap = plt.get_cmap("tab20", len(classes))
    colors: list[tuple[float, float, float, float]] = [cmap(class_index) for class_index in range(len(classes))]
    class_color_map: dict[np.int8, tuple[float, float, float, float]] = dict(zip(classes, colors))
    norm: mcolors.BoundaryNorm = mcolors.BoundaryNorm(boundaries=np.append(classes - 0.5, classes[-1] + 0.5), ncolors=len(classes))
    cmap = mcolors.ListedColormap(colors)
    return class_color_map, norm, cmap


def sub_plot(axes: np.ndarray[Axes] | Axes,
             idx: int,
             img: np.ndarray[np.int8],
             title: str,
             class_color_map: dict[np.int8, tuple[float, float, float, float]],
             cmap_custom: mcolors.ListedColormap,
             norm: mcolors.BoundaryNorm
             ) -> None:
    if type(axes) is Axes:
        axes = np.array([axes], dtype=Axes)
    axes[idx].imshow(img, cmap=cmap_custom, norm=norm)
    legend_patches = [plt.Line2D([idx], [idx], marker='s', color='w', markerfacecolor=color, markersize=10, label=str(cls)) for cls, color in class_color_map.items()]
    axes[idx].legend(handles=legend_patches, title="Rock Labels", bbox_to_anchor=(0, 1), loc='upper left')
    axes[idx].set_title(title)
    axes[idx].set_ylabel("depth (pixels)")
    axes[idx].set_xlabel("horizontal direction (pixels)")


def resize(sample: np.ndarray[np.int8],
           new_shape: tuple[int, int]
           ) -> np.ndarray[np.int8]:
    src_rows, src_cols = sample.shape
    dst_rows, dst_cols = new_shape

    # Compute nearest neighbor indices
    row_indices = np.round(np.linspace(0, src_rows - 1, dst_rows)).astype(int)
    col_indices = np.round(np.linspace(0, src_cols - 1, dst_cols)).astype(int)

    # Use advanced indexing to select nearest neighbors
    resized_array = sample[row_indices[:, None], col_indices]

    return resized_array

def create_random_slice(max_length: int,
                        fraction: float
                        ) -> slice:
    start: int = rd.randint(0, int(max_length * (1 - fraction)))
    return slice(start, start + int(max_length * fraction))

def extract_random_subsection(section: np.ndarray[np.int8],
                              total_pixels_to_keep: int,
                              vertical_fraction: float = 1.,
                              ) -> np.ndarray[np.int8]:
    # Vertical part
    depth_slice: slice = create_random_slice(section.shape[0], vertical_fraction)

    # Horizontal part
    horizontal_fraction: float = total_pixels_to_keep / section.shape[1]
    width_slice: slice = create_random_slice(section.shape[1], horizontal_fraction)

    return section[depth_slice, width_slice], depth_slice, width_slice


def schlumberger_array(nbr_electrodes: int,
                       electrode_locations: np.ndarray[np.float64],
                       data_type: str
                       ) -> list[dc.sources.Dipole]:
    source_list = []
    for a in range(1, nbr_electrodes // 2):
        locations_a = electrode_locations[:(-2 * a) - 1]
        locations_b = electrode_locations[(2 * a) + 1:]
        locations_m = electrode_locations[a:-a - 1]
        locations_n = electrode_locations[a + 1:-a]
        receivers_list_a = [
            dc.receivers.Dipole(locations_m=loc_m, locations_n=loc_n, data_type=data_type)
            for loc_m, loc_n in zip(locations_m, locations_n)
        ]
        source_list_a = [
            dc.sources.Dipole(
                receiver_list_a, location_a=loc_a, location_b=loc_b
            )
            for receiver_list_a, loc_a, loc_b in zip(receivers_list_a, locations_a, locations_b)
        ]
        source_list += source_list_a
    return source_list

def wenner_array(nbr_electrodes: int,
                 electrode_locations: np.ndarray[np.float64],
                 data_type: str
                 ) -> list[dc.sources.Dipole]:
    source_list = []
    for a in range(1, (nbr_electrodes + 3) // 3 + 1):
        locations_a = electrode_locations[:-3 * a:]
        locations_b = electrode_locations[3 * a:]
        locations_m = electrode_locations[a:-2 * a:]
        locations_n = electrode_locations[2 * a:-a]
        receivers_list_a = [
            dc.receivers.Dipole(locations_m=loc_m, locations_n=loc_n, data_type=data_type)
            for loc_m, loc_n in zip(locations_m, locations_n)
        ]
        source_list_a = [
            dc.sources.Dipole(
                receiver_list_a, location_a=loc_a, location_b=loc_b
            )
            for receiver_list_a, loc_a, loc_b in zip(receivers_list_a, locations_a, locations_b)
        ]

        source_list += source_list_a
    return source_list

def compute_active_columns(row: int, is_even_row: bool, total_cols: int, offset: int) -> tuple[int, int]:
    """
    Compute the starting and ending column indices for the given row.
    """
    if is_even_row:
        col_start = math.ceil(row * 1.5) - offset
        col_end = total_cols - math.ceil(row * 1.5) + offset
    else:
        col_start = math.ceil((row + 1) * 1.5) - 1 - offset
        col_end = total_cols - math.ceil((row + 1) * 1.5) + 1 + offset
    return col_start, col_end

def process_pseudo_section_wenner_array(rhoa: list[float],
                                        nbr_electrodes: int
                                        ) -> np.ndarray[np.float64]:
    num_rows: int = (nbr_electrodes - 1) // 3
    num_cols: int = nbr_electrodes - 3

    even_num_cols: bool = (num_cols % 2 == 0)
    if even_num_cols:
        # We want non even number of columns to be able to center the
        # triangle.
        num_cols += 1

    offset: int = (nbr_electrodes - 1) % 2

    pseudo_section: np.ndarray[np.float64] = np.empty(
        (num_rows, num_cols), dtype=np.float64)
    pseudo_section.fill(np.nan)
    value_index: int = 0

    for i in range(num_rows):
        # Determine if the current row is considered "even" based on num_cols
        # parity
        is_even_row = (i % 2 == 0) if even_num_cols else (i % 2 == 1)

        col_start, col_end = compute_active_columns(
            i, is_even_row, num_cols, offset)

        for j in range(col_start, col_end):
            # For even rows, use a special average at the center column
            if is_even_row and j == (num_cols - 1) // 2:
                pseudo_section[i, j] = (
                    rhoa[value_index - 1] + rhoa[value_index]) / 2
            else:
                pseudo_section[i, j] = rhoa[value_index]
                value_index += 1
    return pseudo_section

def process_pseudo_section_schlumberger_array(rhoa: list[float],
                                              nbr_electrodes: int,
                                              ) -> np.ndarray[np.float64]:
    num_cols: int = nbr_electrodes - 3
    num_lines: int = nbr_electrodes // 2 - 1

    pseudo_section: np.ndarray[np.float64] = np.empty(
        (num_lines, num_cols), dtype=np.float64)
    pseudo_section.fill(np.nan)

    value_index: int = 0
    for i in range(num_lines):
        start_col: int = i
        end_col: int = num_cols - i
        num_values_this_row: int = end_col - start_col
        pseudo_section[i, start_col:end_col] = rhoa[value_index: value_index +
                                                    num_values_this_row]
        value_index += num_values_this_row

    return pseudo_section


def process_nky_section(section, nky_accurate, nky_less):
    num_electrodes: int = 24
    vertical_fraction: float = 0.75
    inter_electrode_num_pixels: int = 4
    lateral_padding: int = 4
    array = schlumberger_array

    lateral_padding: int = inter_electrode_num_pixels * lateral_padding
    total_pixels_after_refinement: int = (num_electrodes - 1) * inter_electrode_num_pixels + 2 * lateral_padding

    vertical_size: int = int(vertical_fraction * total_pixels_after_refinement)
    section = resize(section, (vertical_size, total_pixels_after_refinement))

    x_topo = np.linspace(lateral_padding, total_pixels_after_refinement - lateral_padding, total_pixels_after_refinement)
    z_topo = np.zeros_like(x_topo)
    topo_2d = np.c_[x_topo, z_topo]

    electrode_locations_x_simPEG = np.linspace(lateral_padding, total_pixels_after_refinement - lateral_padding, num_electrodes)
    electrode_locations_z_simPEG = np.zeros_like(electrode_locations_x_simPEG)
    electrode_locations_simPEG = np.c_[electrode_locations_x_simPEG, electrode_locations_z_simPEG]
    source_list = array(num_electrodes, electrode_locations_simPEG, "apparent_resistivity")
    survey = dc.Survey(source_list)
    
    rock_classes, inv = np.unique(section, return_inverse=True)
    norm_log_res_values: np.ndarray[np.float64] = np.random.uniform(0, 1, size=len(rock_classes))
    norm_log_resistivity_model = norm_log_res_values[inv].reshape(section.shape)
    resistivity_model = detransform(norm_log_resistivity_model)
    resistivity_model_flat = np.flipud(resistivity_model).ravel()

    hz = [(1, resistivity_model.shape[0])]
    hx = [(1, resistivity_model.shape[1])]
    mesh = TensorMesh((hx, hz), origin="0N")

    active_cells = active_from_xyz(mesh, topo_2d)

    resistivity_map = maps.IdentityMap(mesh)
    survey.drape_electrodes_on_topography(mesh, active_cells, option="top", topography=topo_2d)
    _ = survey.set_geometric_factor()

    accurate_resistivity_simulation = dc.simulation_2d.Simulation2DNodal(mesh=mesh, survey=survey, rhoMap=resistivity_map, nky=nky_accurate)
    resistivity_simulation = dc.simulation_2d.Simulation2DNodal(mesh=mesh, survey=survey, rhoMap=resistivity_map, nky=nky_less)

    accurate_fields_start = perf_counter()
    accurate_fields = accurate_resistivity_simulation.fields(resistivity_model_flat)
    accurate_fields_end = perf_counter()
    accurate_fields_time = accurate_fields_end - accurate_fields_start
    fields_start = perf_counter()
    fields = resistivity_simulation.fields(resistivity_model_flat)
    fields_end = perf_counter()
    fields_time = fields_end - fields_start

    accurate_pseudosection_start = perf_counter()
    accurate_pseudosection_flat = accurate_resistivity_simulation.dpred(resistivity_model_flat, accurate_fields)
    accurate_pseudosection_end = perf_counter()
    accurate_pseudosection_time = accurate_pseudosection_end - accurate_pseudosection_start
    pseudosection_start = perf_counter()
    pseudosection_flat = resistivity_simulation.dpred(resistivity_model_flat, fields)
    pseudosection_end = perf_counter()
    pseudosection_time = pseudosection_end - pseudosection_start

    accurate_J_start = perf_counter()
    accurate_J = accurate_resistivity_simulation.getJ(resistivity_model_flat, accurate_fields)
    accurate_J_end = perf_counter()
    accurate_J_time = accurate_J_end - accurate_J_start
    J_start = perf_counter()
    J = resistivity_simulation.getJ(resistivity_model_flat, fields)
    J_end = perf_counter()
    J_time = J_end - J_start

    accurate_JtJ_diag = np.flipud(np.sqrt(np.sum(np.square(accurate_J), axis=0)).reshape(resistivity_model.shape))
    JtJ_diag = np.flipud(np.sqrt(np.sum(np.square(J), axis=0)).reshape(resistivity_model.shape))

    if array == wenner_array:
        process_pseudo_section = process_pseudo_section_wenner_array
    else:
        process_pseudo_section = process_pseudo_section_schlumberger_array

    accurate_pseudosection = process_pseudo_section(accurate_pseudosection_flat, num_electrodes)
    pseudosection = process_pseudo_section(pseudosection_flat, num_electrodes)

    times = {
        "accurate_fields": accurate_fields_time,
        "fields": fields_time,
        "accurate_pseudosection": accurate_pseudosection_time,
        "pseudosection": pseudosection_time,
        "accurate_J": accurate_J_time,
        "J": J_time,
    }

    return accurate_pseudosection, pseudosection, accurate_JtJ_diag, JtJ_diag, norm_log_resistivity_model, times

def process_air_section(section):
    num_electrodes: int = 24
    vertical_fraction: float = 0.75
    inter_electrode_num_pixels: int = 4
    lateral_padding: int = 4
    array = schlumberger_array

    air_layer_thickness = 10
    air_resistivity = 1e8

    lateral_padding: int = inter_electrode_num_pixels * lateral_padding
    total_pixels_after_refinement: int = (num_electrodes - 1) * inter_electrode_num_pixels + 2 * lateral_padding

    vertical_size: int = int(vertical_fraction * total_pixels_after_refinement)
    section = resize(section, (vertical_size, total_pixels_after_refinement))

    x_topo = np.linspace(lateral_padding, total_pixels_after_refinement - lateral_padding, total_pixels_after_refinement)
    z_topo = np.zeros_like(x_topo)
    topo_2d = np.c_[x_topo, z_topo]

    electrode_locations_x_simPEG = np.linspace(lateral_padding, total_pixels_after_refinement - lateral_padding, num_electrodes)
    electrode_locations_z_simPEG = np.zeros_like(electrode_locations_x_simPEG)
    electrode_locations_simPEG = np.c_[electrode_locations_x_simPEG, electrode_locations_z_simPEG]
    source_list = array(num_electrodes, electrode_locations_simPEG, "apparent_resistivity")

    air_survey = dc.Survey(source_list)
    no_air_survey = dc.Survey(source_list)

    
    rock_classes, inv = np.unique(section, return_inverse=True)
    norm_log_res_values: np.ndarray[np.float64] = np.random.uniform(0, 1, size=len(rock_classes))
    norm_log_resistivity_model = norm_log_res_values[inv].reshape(section.shape)
    resistivity_model = detransform(norm_log_resistivity_model)
    resistivity_model_flat = np.flipud(resistivity_model).ravel()

    air_layer = np.full((air_layer_thickness, resistivity_model.shape[1]), air_resistivity, dtype=float)
    air_resistivity_model = np.vstack((air_layer, resistivity_model))

    hz = [(1, air_resistivity_model.shape[0])]
    hx = [(1, air_resistivity_model.shape[1])]
    air_mesh = TensorMesh((hx, hz), origin="0N")
    air_mesh.origin += [0, air_layer_thickness]

    hz = [(1, resistivity_model.shape[0])]
    hx = [(1, resistivity_model.shape[1])]
    no_air_mesh = TensorMesh((hx, hz), origin="0N")

    air_active_cells = active_from_xyz(air_mesh, topo_2d)
    no_air_active_cells = active_from_xyz(no_air_mesh, topo_2d)

    air_resistivity_map = maps.InjectActiveCells(air_mesh, air_active_cells, air_resistivity)
    no_air_resistivity_map = maps.IdentityMap(no_air_mesh)

    air_survey.drape_electrodes_on_topography(air_mesh, air_active_cells, option="top", topography=topo_2d)
    no_air_survey.drape_electrodes_on_topography(no_air_mesh, no_air_active_cells, option="top", topography=topo_2d)
    _ = air_survey.set_geometric_factor()
    _ = no_air_survey.set_geometric_factor()

    air_resistivity_simulation = dc.simulation_2d.Simulation2DNodal(mesh=air_mesh, survey=air_survey, rhoMap=air_resistivity_map, nky=5)
    no_air_resistivity_simulation = dc.simulation_2d.Simulation2DNodal(mesh=no_air_mesh, survey=no_air_survey, rhoMap=no_air_resistivity_map, nky=5)

    air_fields_start = perf_counter()
    air_fields = air_resistivity_simulation.fields(resistivity_model_flat)
    air_fields_end = perf_counter()
    air_fields_time = air_fields_end - air_fields_start
    no_air_fields_start = perf_counter()
    no_air_fields = no_air_resistivity_simulation.fields(resistivity_model_flat)
    no_air_fields_end = perf_counter()
    no_air_fields_time = no_air_fields_end - no_air_fields_start

    air_pseudosection_start = perf_counter()
    air_pseudosection_flat = air_resistivity_simulation.dpred(resistivity_model_flat, air_fields)
    air_pseudosection_end = perf_counter()
    air_pseudosection_time = air_pseudosection_end - air_pseudosection_start
    no_air_pseudosection_start = perf_counter()
    no_air_pseudosection_flat = no_air_resistivity_simulation.dpred(resistivity_model_flat, no_air_fields)
    no_air_pseudosection_end = perf_counter()
    no_air_pseudosection_time = no_air_pseudosection_end - no_air_pseudosection_start

    air_J_start = perf_counter()
    air_J = air_resistivity_simulation.getJ(resistivity_model_flat, air_fields)
    air_J_end = perf_counter()
    air_J_time = air_J_end - air_J_start
    no_air_J_start = perf_counter()
    no_air_J = no_air_resistivity_simulation.getJ(resistivity_model_flat, no_air_fields)
    no_air_J_end = perf_counter()
    no_air_J_time = no_air_J_end - no_air_J_start

    air_JtJ_diag = np.flipud(np.sqrt(np.sum(np.square(air_J), axis=0)).reshape(resistivity_model.shape))
    no_air_JtJ_diag = np.flipud(np.sqrt(np.sum(np.square(no_air_J), axis=0)).reshape(resistivity_model.shape))

    if array == wenner_array:
        process_pseudo_section = process_pseudo_section_wenner_array
    else:
        process_pseudo_section = process_pseudo_section_schlumberger_array

    air_pseudosection = process_pseudo_section(air_pseudosection_flat, num_electrodes)
    no_air_pseudosection = process_pseudo_section(no_air_pseudosection_flat, num_electrodes)

    times = {
        "air_fields": air_fields_time,
        "no_air_fields": no_air_fields_time,
        "air_pseudosection": air_pseudosection_time,
        "no_air_pseudosection": no_air_pseudosection_time,
        "air_J": air_J_time,
        "no_air_J": no_air_J_time,
    }

    return air_pseudosection, no_air_pseudosection, air_JtJ_diag, no_air_JtJ_diag, norm_log_resistivity_model, times


def process_exp_section(section):
    num_electrodes: int = 24
    vertical_fraction: float = 0.75
    inter_electrode_num_pixels: int = 4
    lateral_padding: int = 4
    array = schlumberger_array

    lateral_padding: int = inter_electrode_num_pixels * lateral_padding
    total_pixels_after_refinement: int = (num_electrodes - 1) * inter_electrode_num_pixels + 2 * lateral_padding

    vertical_size: int = int(vertical_fraction * total_pixels_after_refinement)
    section = resize(section, (vertical_size, total_pixels_after_refinement))

    x_topo = np.linspace(lateral_padding, total_pixels_after_refinement - lateral_padding, total_pixels_after_refinement)
    z_topo = np.zeros_like(x_topo)
    topo_2d = np.c_[x_topo, z_topo]

    electrode_locations_x_simPEG = np.linspace(lateral_padding, total_pixels_after_refinement - lateral_padding, num_electrodes)
    electrode_locations_z_simPEG = np.zeros_like(electrode_locations_x_simPEG)
    electrode_locations_simPEG = np.c_[electrode_locations_x_simPEG, electrode_locations_z_simPEG]
    source_list = array(num_electrodes, electrode_locations_simPEG, "apparent_resistivity")
    survey = dc.Survey(source_list)
    
    rock_classes, inv = np.unique(section, return_inverse=True)
    norm_log_res_values: np.ndarray[np.float64] = np.random.uniform(0, 1, size=len(rock_classes))
    norm_log_resistivity_model = norm_log_res_values[inv].reshape(section.shape)
    log_resistivity_model = np.log(detransform(norm_log_resistivity_model))
    log_resistivity_model_flat = np.flipud(log_resistivity_model).ravel()
    resistivity_model_flat = np.flipud(detransform(norm_log_resistivity_model)).ravel()

    hz = [(1, log_resistivity_model.shape[0])]
    hx = [(1, log_resistivity_model.shape[1])]
    mesh = TensorMesh((hx, hz), origin="0N")

    active_cells = active_from_xyz(mesh, topo_2d)

    exp_resistivity_map = maps.IdentityMap(mesh) * maps.ExpMap(mesh)
    resistivity_map = maps.IdentityMap(mesh)
    survey.drape_electrodes_on_topography(mesh, active_cells, option="top", topography=topo_2d)
    _ = survey.set_geometric_factor()

    exp_resistivity_simulation = dc.simulation_2d.Simulation2DNodal(mesh=mesh, survey=survey, rhoMap=exp_resistivity_map, nky=5)
    resistivity_simulation = dc.simulation_2d.Simulation2DNodal(mesh=mesh, survey=survey, rhoMap=resistivity_map, nky=5)

    exp_fields_start = perf_counter()
    exp_fields = exp_resistivity_simulation.fields(log_resistivity_model_flat)
    exp_fields_end = perf_counter()
    exp_fields_time = exp_fields_end - exp_fields_start
    fields_start = perf_counter()
    fields = resistivity_simulation.fields(resistivity_model_flat)
    fields_end = perf_counter()
    fields_time = fields_end - fields_start

    exp_pseudosection_start = perf_counter()
    exp_pseudosection_flat = exp_resistivity_simulation.dpred(log_resistivity_model_flat, exp_fields)
    exp_pseudosection_end = perf_counter()
    exp_pseudosection_time = exp_pseudosection_end - exp_pseudosection_start
    pseudosection_start = perf_counter()
    pseudosection_flat = resistivity_simulation.dpred(resistivity_model_flat, fields)
    pseudosection_end = perf_counter()
    pseudosection_time = pseudosection_end - pseudosection_start

    exp_J_start = perf_counter()
    exp_J = exp_resistivity_simulation.getJ(log_resistivity_model_flat, exp_fields)
    exp_J_end = perf_counter()
    exp_J_time = exp_J_end - exp_J_start
    J_start = perf_counter()
    J = resistivity_simulation.getJ(resistivity_model_flat, fields)
    J_end = perf_counter()
    J_time = J_end - J_start

    exp_JtJ_diag = np.flipud(np.sqrt(np.sum(np.square(exp_J), axis=0)).reshape(log_resistivity_model.shape))
    JtJ_diag = np.flipud(np.sqrt(np.sum(np.square(J), axis=0)).reshape(log_resistivity_model.shape)) # log_resistivity_model and resistivity model have the same shape.

    if array == wenner_array:
        process_pseudo_section = process_pseudo_section_wenner_array
    else:
        process_pseudo_section = process_pseudo_section_schlumberger_array

    exp_pseudosection = process_pseudo_section(exp_pseudosection_flat, num_electrodes)
    pseudosection = process_pseudo_section(pseudosection_flat, num_electrodes)

    times = {
        "exp_fields": exp_fields_time,
        "fields": fields_time,
        "exp_pseudosection": exp_pseudosection_time,
        "pseudosection": pseudosection_time,
        "exp_J": exp_J_time,
        "J": J_time,
    }

    return exp_pseudosection, pseudosection, exp_JtJ_diag, JtJ_diag, norm_log_resistivity_model, times
    


def air_test(NUM_SAMPLES: int, dataset_path: Path, samples_per_npz: int, num_files: int, SAVE_FOLDER: Path) -> None:
    save_dir = SAVE_FOLDER / "air_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    air_pseudosections = []
    no_air_pseudosections = []
    air_JtJ_diags = []
    no_air_JtJ_diags = []
    norm_log_resistivity_models = []
    times_dict = {
        "air_fields": [],
        "no_air_fields": [],
        "air_pseudosection": [],
        "no_air_pseudosection": [],
        "air_J": [],
        "no_air_J": [],
    }
    npz_index = 0
    multi_array: np.ndarray[np.int8] = np.load(dataset_path / f"{npz_index}.npz")["arr_0"]
    for i in tqdm(range(1, NUM_SAMPLES + 1), desc="Processing sections for air test", unit="section"):
        if i % samples_per_npz:
            npz_index += 1
            npz_index %= num_files
            multi_array: np.ndarray[np.int8] = np.load(dataset_path / f"{npz_index}.npz")["arr_0"]
        section: np.ndarray[np.int8] = multi_array[rd.randint(0, len(multi_array) - 1)]
        results = process_air_section(section)
        air_pseudosections.append(results[0])
        no_air_pseudosections.append(results[1])
        air_JtJ_diags.append(results[2])
        no_air_JtJ_diags.append(results[3])
        norm_log_resistivity_models.append(results[-2])
        for key in times_dict.keys():
            times_dict[key].append(results[-1][key])
    
    air_pseudosections = np.array(air_pseudosections)
    no_air_pseudosections = np.array(no_air_pseudosections)
    norm_log_resistivity_models = np.array(norm_log_resistivity_models)
    air_JtJ_diags = np.array(air_JtJ_diags)
    no_air_JtJ_diags = np.array(no_air_JtJ_diags)
    times_dict = {key: np.array(value) for key, value in times_dict.items()}

    for key, value in times_dict.items():
        print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f} seconds")

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    im0 = axes[0, 0].imshow(norm_log_resistivity_models.mean(axis=0), cmap="viridis")
    axes[0, 0].set_title("Mean Log Resistivity Model")
    axes[0, 0].set_xlabel("horizontal direction (pixels)")
    axes[0, 0].set_ylabel("depth (pixels)")
    cbar0 = fig.colorbar(im0, ax=axes[0, 0], orientation='vertical', fraction=0.025, pad=0.02)
    cbar0.set_label(r"Resistivity ($\Omega$m)", rotation=270, labelpad=15)

    mean_air_pseudosection = air_pseudosections.mean(axis=0)
    mean_no_air_pseudosection = no_air_pseudosections.mean(axis=0)
    std_air_pseudosection = air_pseudosections.std(axis=0)
    std_no_air_pseudosection = no_air_pseudosections.std(axis=0)
    vmin = min(np.nanmin(mean_air_pseudosection), np.nanmin(mean_no_air_pseudosection), 
               np.nanmin(std_air_pseudosection), np.nanmin(std_no_air_pseudosection))
    vmax = max(np.nanmax(mean_air_pseudosection), np.nanmax(mean_no_air_pseudosection),
               np.nanmax(std_air_pseudosection), np.nanmax(std_no_air_pseudosection))
    norm_pseudo = LogNorm(vmin=vmin, vmax=vmax)

    im1 = axes[0, 1].imshow(mean_air_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[0, 1].set_title("Mean Pseudosection")
    axes[0, 1].set_xlabel("horizontal direction")
    axes[0, 1].set_ylabel("pseudodepth")
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], orientation='vertical', fraction=0.025, pad=0.02)
    cbar1.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=15)

    im2 = axes[0, 2].imshow(mean_no_air_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[0, 2].set_title("Mean Pseudosection (without air layer)")
    axes[0, 2].set_xlabel("horizontal direction")
    axes[0, 2].set_ylabel("pseudodepth")
    cbar2 = fig.colorbar(im2, ax=axes[0, 2], orientation='vertical', fraction=0.025, pad=0.02)
    cbar2.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=15)

    mean_air_JtJ_diag = air_JtJ_diags.mean(axis=0)
    mean_no_air_JtJ_diag = no_air_JtJ_diags.mean(axis=0)
    std_air_JtJ_diag = air_JtJ_diags.std(axis=0)
    std_no_air_JtJ_diag = no_air_JtJ_diags.std(axis=0)
    vmin = min(np.nanmin(mean_air_JtJ_diag), np.nanmin(mean_no_air_JtJ_diag),
               np.nanmin(std_air_JtJ_diag), np.nanmin(std_no_air_JtJ_diag))
    vmax = max(np.nanmax(mean_air_JtJ_diag), np.nanmax(mean_no_air_JtJ_diag),
               np.nanmax(std_air_JtJ_diag), np.nanmax(std_no_air_JtJ_diag))
    norm_sensi = LogNorm(vmin=vmin, vmax=vmax)
    im3 = axes[0, 3].imshow(mean_air_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[0, 3].set_title("Mean sensitivity")
    axes[0, 3].set_xlabel("horizontal direction (pixels)")
    axes[0, 3].set_ylabel("depth (pixels)")
    cbar3 = fig.colorbar(im3, ax=axes[0, 3], orientation='vertical', fraction=0.025, pad=0.02)
    cbar3.set_label("Sensitivty", rotation=270, labelpad=15)

    im4 = axes[0, 4].imshow(mean_no_air_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[0, 4].set_title("Mean sensitivity (without air layer)")
    axes[0, 4].set_xlabel("horizontal direction (pixels)")
    axes[0, 4].set_ylabel("depth (pixels)")
    cbar4 = fig.colorbar(im4, ax=axes[0, 4], orientation='vertical', fraction=0.025, pad=0.02)
    cbar4.set_label("Sensitivty", rotation=270, labelpad=15)

    im5 = axes[1, 0].imshow(norm_log_resistivity_models.std(axis=0), cmap="viridis")
    axes[1, 0].set_title("Std Log Resistivity Model")
    axes[1, 0].set_xlabel("horizontal direction (pixels)")
    axes[1, 0].set_ylabel("depth (pixels)")
    cbar5 = fig.colorbar(im5, ax=axes[1, 0], orientation='vertical', fraction=0.025, pad=0.02)
    cbar5.set_label(r"Resistivity ($\Omega$m)", rotation=270, labelpad=15)
    
    im6 = axes[1, 1].imshow(std_air_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[1, 1].set_title("Std Pseudosection")
    axes[1, 1].set_xlabel("horizontal direction")
    axes[1, 1].set_ylabel("pseudodepth")
    cbar6 = fig.colorbar(im6, ax=axes[1, 1], orientation='vertical', fraction=0.025, pad=0.02)
    cbar6.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=15)

    im7 = axes[1, 2].imshow(std_no_air_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[1, 2].set_title("Std Sensitivity Pseudosection (without air layer)")
    axes[1, 2].set_xlabel("horizontal direction")
    axes[1, 2].set_ylabel("pseudodepth")
    cbar7 = fig.colorbar(im7, ax=axes[1, 2], orientation='vertical', fraction=0.025, pad=0.02)
    cbar7.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=15)

    im8 = axes[1, 3].imshow(std_air_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[1, 3].set_title("Std Sensitivity")
    axes[1, 3].set_xlabel("horizontal direction (pixels)")
    axes[1, 3].set_ylabel("depth (pixels)")
    cbar8 = fig.colorbar(im8, ax=axes[1, 3], orientation='vertical', fraction=0.025, pad=0.02)
    cbar8.set_label("Sensitivity", rotation=270, labelpad=15)

    im9 = axes[1, 4].imshow(std_no_air_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[1, 4].set_title("Std Sensitivity (without air layer)")
    axes[1, 4].set_xlabel("horizontal direction (pixels)")
    axes[1, 4].set_ylabel("depth (pixels)")
    cbar9 = fig.colorbar(im9, ax=axes[1, 4], orientation='vertical', fraction=0.025, pad=0.02)
    cbar9.set_label("Sensitivity", rotation=270, labelpad=15)

    plt.suptitle(f"Air Test Results ({NUM_SAMPLES} samples)")
    plt.tight_layout()
    plt.savefig(save_dir / "air_test_means_std.png")
    plt.close(fig)

    mean_square_relative_difference_pseudosection = np.mean(np.square(air_pseudosections - no_air_pseudosections) / no_air_pseudosections, axis=0)
    mean_square_relative_difference_JtJ = np.mean(np.square(air_JtJ_diags - no_air_JtJ_diags) / no_air_JtJ_diags, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    vmin = np.nanmin(mean_square_relative_difference_pseudosection)
    vmax = np.nanmax(mean_square_relative_difference_pseudosection)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im0 = axes[0].imshow(mean_square_relative_difference_pseudosection, cmap="viridis", norm=norm)
    axes[0].set_title("Mean Square Relative Difference Pseudosection")
    axes[0].set_xlabel("horizontal direction")
    axes[0].set_ylabel("pseudodepth")
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.025, pad=0.02)
    cbar0.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=15)

    vmin = np.min(mean_square_relative_difference_JtJ)
    vmax = np.max(mean_square_relative_difference_JtJ)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im1 = axes[1].imshow(mean_square_relative_difference_JtJ, cmap="viridis", norm=norm)
    axes[1].set_title("Mean Square Relative Difference Sensitivity")
    axes[1].set_xlabel("horizontal direction (pixels)")
    axes[1].set_ylabel("depth (pixels)")
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.025, pad=0.02)
    cbar1.set_label("Sensitivity", rotation=270, labelpad=15)

    plt.suptitle(f"Air Test Mean Square Relative Difference ({NUM_SAMPLES} samples)")
    plt.tight_layout()
    plt.savefig(save_dir / "air_test_mean_square_relative_difference.png")
    plt.close(fig)


def expMap_test(NUM_SAMPLES: int, dataset_path: Path, samples_per_npz: int, num_files: int, SAVE_FOLDER: Path) -> None:
    save_dir = SAVE_FOLDER / "expMap_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    exp_pseudosections = []
    pseudosections = []
    exp_JtJ_diags = []
    JtJ_diags = []
    norm_log_resistivity_models = []
    times_dict = {
        "exp_fields": [],
        "fields": [],
        "exp_pseudosection": [],
        "pseudosection": [],
        "exp_J": [],
        "J": [],
    }
    npz_index = 0
    multi_array: np.ndarray[np.int8] = np.load(dataset_path / f"{npz_index}.npz")["arr_0"]
    for i in tqdm(range(1, NUM_SAMPLES + 1), desc="Processing sections for nky test", unit="section"):
        if i % samples_per_npz:
            npz_index += 1
            npz_index %= num_files
            multi_array: np.ndarray[np.int8] = np.load(dataset_path / f"{npz_index}.npz")["arr_0"]
        section: np.ndarray[np.int8] = multi_array[rd.randint(0, len(multi_array) - 1)]
        results = process_exp_section(section)
        exp_pseudosections.append(results[0])
        pseudosections.append(results[1])
        exp_JtJ_diags.append(results[2])
        JtJ_diags.append(results[3])
        norm_log_resistivity_models.append(results[-2])
        for key in times_dict.keys():
            times_dict[key].append(results[-1][key])
        
    exp_pseudosections = np.array(exp_pseudosections)
    pseudosections = np.array(pseudosections)
    norm_log_resistivity_models = np.array(norm_log_resistivity_models)
    exp_JtJ_diags = np.array(exp_JtJ_diags)
    JtJ_diags = np.array(JtJ_diags)
    times_dict = {key: np.array(value) for key, value in times_dict.items()}
    for key, value in times_dict.items():
        print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f} seconds")

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    im0 = axes[0, 0].imshow(norm_log_resistivity_models.mean(axis=0), cmap="viridis")
    axes[0, 0].set_title("Mean Log Resistivity Model")
    axes[0, 0].set_xlabel("horizontal direction (pixels)")
    axes[0, 0].set_ylabel("depth (pixels)")
    cbar0 = fig.colorbar(im0, ax=axes[0, 0], orientation='vertical', fraction=0.025, pad=0.02)
    cbar0.set_label(r"Resistivity ($\Omega$m)", rotation=270, labelpad=15)
    
    mean_exp_pseudosection = exp_pseudosections.mean(axis=0)
    mean_pseudosection = pseudosections.mean(axis=0)
    std_exp_pseudosection = exp_pseudosections.std(axis=0)
    std_pseudosection = pseudosections.std(axis=0)
    vmin = min(np.nanmin(mean_exp_pseudosection), np.nanmin(mean_pseudosection), 
               np.nanmin(std_exp_pseudosection), np.nanmin(std_pseudosection))
    vmax = max(np.nanmax(mean_exp_pseudosection), np.nanmax(mean_pseudosection),
               np.nanmax(std_exp_pseudosection), np.nanmax(std_pseudosection))
    norm_pseudo = LogNorm(vmin=vmin, vmax=vmax)
    im1 = axes[0, 1].imshow(mean_exp_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[0, 1].set_title(f"Mean Pseudosection (log model)")
    axes[0, 1].set_xlabel("horizontal direction")
    axes[0, 1].set_ylabel("pseudodepth")
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], orientation='vertical', fraction=0.025, pad=0.02)
    cbar1.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=15)
    
    im2 = axes[0, 2].imshow(mean_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[0, 2].set_title(f"Mean Pseudosection")
    axes[0, 2].set_xlabel("horizontal direction")
    axes[0, 2].set_ylabel("pseudodepth")
    cbar2 = fig.colorbar(im2, ax=axes[0, 2], orientation='vertical', fraction=0.025, pad=0.02)
    cbar2.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=0)

    mean_exp_JtJ_diag = exp_JtJ_diags.mean(axis=0)
    mean_JtJ_diag = JtJ_diags.mean(axis=0)
    std_exp_JtJ_diag = exp_JtJ_diags.std(axis=0)
    std_JtJ_diag = JtJ_diags.std(axis=0)
    vmin = min(np.nanmin(mean_exp_JtJ_diag), np.nanmin(mean_JtJ_diag),
               np.nanmin(std_exp_JtJ_diag), np.nanmin(std_JtJ_diag))
    vmax = max(np.nanmax(mean_exp_JtJ_diag), np.nanmax(mean_JtJ_diag),
               np.nanmax(std_exp_JtJ_diag), np.nanmax(std_JtJ_diag))
    norm_sensi = LogNorm(vmin=vmin, vmax=vmax)
    im3 = axes[0, 3].imshow(mean_exp_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[0, 3].set_title(f"Mean sensitivity (log_model)")
    axes[0, 3].set_xlabel("horizontal direction (pixels)")
    axes[0, 3].set_ylabel("depth (pixels)")
    cbar3 = fig.colorbar(im3, ax=axes[0, 3], orientation='vertical', fraction=0.025, pad=0.02)
    cbar3.set_label("Sensitivity", rotation=270, labelpad=15)

    im4 = axes[0, 4].imshow(mean_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[0, 4].set_title(f"Mean sensitivity")
    axes[0, 4].set_xlabel("horizontal direction (pixels)")
    axes[0, 4].set_ylabel("depth (pixels)")
    cbar4 = fig.colorbar(im4, ax=axes[0, 4], orientation='vertical', fraction=0.025, pad=0.02)
    cbar4.set_label("Sensitivity", rotation=270, labelpad=15)

    im5 = axes[1, 0].imshow(norm_log_resistivity_models.std(axis=0), cmap="viridis")
    axes[1, 0].set_title("Std Log Resistivity Model")
    axes[1, 0].set_xlabel("horizontal direction (pixels)")
    axes[1, 0].set_ylabel("depth (pixels)")
    cbar5 = fig.colorbar(im5, ax=axes[1, 0], orientation='vertical', fraction=0.025, pad=0.02)
    cbar5.set_label(r"Resistivity ($\Omega$m)", rotation=270, labelpad=15)

    im6 = axes[1, 1].imshow(std_exp_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[1, 1].set_title(f"Std Pseudosection (log model)")
    axes[1, 1].set_xlabel("horizontal direction")
    axes[1, 1].set_ylabel("pseudodepth")
    cbar6 = fig.colorbar(im6, ax=axes[1, 1], orientation='vertical', fraction=0.025, pad=0.02)
    cbar6.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=0)

    im7 = axes[1, 2].imshow(std_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[1, 2].set_title(f"Std Pseudosection")
    axes[1, 2].set_xlabel("horizontal direction")
    axes[1, 2].set_ylabel("pseudodepth")
    cbar7 = fig.colorbar(im7, ax=axes[1, 2], orientation='vertical', fraction=0.025, pad=0.02)
    cbar7.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=0)

    im8 = axes[1, 3].imshow(std_exp_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[1, 3].set_title(f"Std Sensitivity (log model)")
    axes[1, 3].set_xlabel("horizontal direction (pixels)")
    axes[1, 3].set_ylabel("depth (pixels)")
    cbar8 = fig.colorbar(im8, ax=axes[1, 3], orientation='vertical', fraction=0.025, pad=0.02)
    cbar8.set_label("Sensitivity", rotation=270, labelpad=15)

    im9 = axes[1, 4].imshow(std_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[1, 4].set_title(f"Std Sensitivity")
    axes[1, 4].set_xlabel("horizontal direction (pixels)")
    axes[1, 4].set_ylabel("depth (pixels)")
    cbar9 = fig.colorbar(im9, ax=axes[1, 4], orientation='vertical', fraction=0.025, pad=0.02)
    cbar9.set_label("Sensitivity", rotation=270, labelpad=15)
    plt.suptitle(f"Exponential Map Test Results ({NUM_SAMPLES} samples)")
    plt.tight_layout()
    plt.savefig(save_dir / f"exp_test_means_std.png")
    plt.close(fig)

    mean_square_relative_difference_pseudosection = np.mean(np.square(exp_pseudosections - pseudosections) / exp_pseudosections, axis=0)
    mean_square_relative_difference_JtJ = np.mean(np.square(exp_JtJ_diags - JtJ_diags) / exp_JtJ_diags, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    vmin = np.nanmin(mean_square_relative_difference_pseudosection)
    vmax = np.nanmax(mean_square_relative_difference_pseudosection)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im0 = axes[0].imshow(mean_square_relative_difference_pseudosection, cmap="viridis", norm=norm)
    axes[0].set_title("Mean Square Relative Difference Pseudosection")
    axes[0].set_xlabel("horizontal direction")
    axes[0].set_ylabel("pseudodepth")
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.025, pad=0.02)
    cbar0.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=15)
    vmin = np.min(mean_square_relative_difference_JtJ)
    vmax = np.max(mean_square_relative_difference_JtJ)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im1 = axes[1].imshow(mean_square_relative_difference_JtJ, cmap="viridis", norm=norm)
    axes[1].set_title("Mean Square Relative Difference Sensitivity")
    axes[1].set_xlabel("horizontal direction (pixels)")
    axes[1].set_ylabel("depth (pixels)")
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.025, pad=0.02)
    cbar1.set_label("Sensitivity", rotation=270, labelpad=15)
    plt.suptitle(f"Exponential Map Test Mean Square Relative Difference ({NUM_SAMPLES} samples)")
    plt.tight_layout()
    plt.savefig(save_dir / f"exp_test_mean_square_relative_difference.png")
    plt.close(fig)



def nky_test(NUM_SAMPLES: int, dataset_path: Path, samples_per_npz: int, num_files: int, SAVE_FOLDER: Path) -> None:
    save_dir = SAVE_FOLDER / "nky_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    nky_accurate = 11
    nky_less = 5

    accurate_pseudosections = []
    pseudosections = []
    accurate_JtJ_diags = []
    JtJ_diags = []
    norm_log_resistivity_models = []
    times_dict = {
        "accurate_fields": [],
        "fields": [],
        "accurate_pseudosection": [],
        "pseudosection": [],
        "accurate_J": [],
        "J": [],
    }
    npz_index = 0
    multi_array: np.ndarray[np.int8] = np.load(dataset_path / f"{npz_index}.npz")["arr_0"]
    for i in tqdm(range(1, NUM_SAMPLES + 1), desc="Processing sections for nky test", unit="section"):
        if i % samples_per_npz:
            npz_index += 1
            npz_index %= num_files
            multi_array: np.ndarray[np.int8] = np.load(dataset_path / f"{npz_index}.npz")["arr_0"]
        section: np.ndarray[np.int8] = multi_array[rd.randint(0, len(multi_array) - 1)]
        results = process_nky_section(section, nky_accurate, nky_less)
        accurate_pseudosections.append(results[0])
        pseudosections.append(results[1])
        accurate_JtJ_diags.append(results[2])
        JtJ_diags.append(results[3])
        norm_log_resistivity_models.append(results[-2])
        for key in times_dict.keys():
            times_dict[key].append(results[-1][key])
        
    accurate_pseudosections = np.array(accurate_pseudosections)
    pseudosections = np.array(pseudosections)
    norm_log_resistivity_models = np.array(norm_log_resistivity_models)
    accurate_JtJ_diags = np.array(accurate_JtJ_diags)
    JtJ_diags = np.array(JtJ_diags)
    times_dict = {key: np.array(value) for key, value in times_dict.items()}
    for key, value in times_dict.items():
        print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f} seconds")

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    im0 = axes[0, 0].imshow(norm_log_resistivity_models.mean(axis=0), cmap="viridis")
    axes[0, 0].set_title("Mean Log Resistivity Model")
    axes[0, 0].set_xlabel("horizontal direction (pixels)")
    axes[0, 0].set_ylabel("depth (pixels)")
    cbar0 = fig.colorbar(im0, ax=axes[0, 0], orientation='vertical', fraction=0.025, pad=0.02)
    cbar0.set_label(r"Resistivity ($\Omega$m)", rotation=270, labelpad=15)
    
    mean_accurate_pseudosection = accurate_pseudosections.mean(axis=0)
    mean_pseudosection = pseudosections.mean(axis=0)
    std_accurate_pseudosection = accurate_pseudosections.std(axis=0)
    std_pseudosection = pseudosections.std(axis=0)
    vmin = min(np.nanmin(mean_accurate_pseudosection), np.nanmin(mean_pseudosection), 
               np.nanmin(std_accurate_pseudosection), np.nanmin(std_pseudosection))
    vmax = max(np.nanmax(mean_accurate_pseudosection), np.nanmax(mean_pseudosection),
               np.nanmax(std_accurate_pseudosection), np.nanmax(std_pseudosection))
    norm_pseudo = LogNorm(vmin=vmin, vmax=vmax)
    im1 = axes[0, 1].imshow(mean_accurate_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[0, 1].set_title(f"Mean Pseudosection (nky={nky_accurate})")
    axes[0, 1].set_xlabel("horizontal direction")
    axes[0, 1].set_ylabel("pseudodepth")
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], orientation='vertical', fraction=0.025, pad=0.02)
    cbar1.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=15)
    
    im2 = axes[0, 2].imshow(mean_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[0, 2].set_title(f"Mean Pseudosection (nky={nky_less})")
    axes[0, 2].set_xlabel("horizontal direction")
    axes[0, 2].set_ylabel("pseudodepth")
    cbar2 = fig.colorbar(im2, ax=axes[0, 2], orientation='vertical', fraction=0.025, pad=0.02)
    cbar2.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=0)

    mean_accurate_JtJ_diag = accurate_JtJ_diags.mean(axis=0)
    mean_JtJ_diag = JtJ_diags.mean(axis=0)
    std_accurate_JtJ_diag = accurate_JtJ_diags.std(axis=0)
    std_JtJ_diag = JtJ_diags.std(axis=0)
    vmin = min(np.nanmin(mean_accurate_JtJ_diag), np.nanmin(mean_JtJ_diag),
               np.nanmin(std_accurate_JtJ_diag), np.nanmin(std_JtJ_diag))
    vmax = max(np.nanmax(mean_accurate_JtJ_diag), np.nanmax(mean_JtJ_diag),
               np.nanmax(std_accurate_JtJ_diag), np.nanmax(std_JtJ_diag))
    norm_sensi = LogNorm(vmin=vmin, vmax=vmax)
    im3 = axes[0, 3].imshow(mean_accurate_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[0, 3].set_title(f"Mean sensitivity (nky={nky_accurate})")
    axes[0, 3].set_xlabel("horizontal direction (pixels)")
    axes[0, 3].set_ylabel("depth (pixels)")
    cbar3 = fig.colorbar(im3, ax=axes[0, 3], orientation='vertical', fraction=0.025, pad=0.02)
    cbar3.set_label("Sensitivity", rotation=270, labelpad=15)

    im4 = axes[0, 4].imshow(mean_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[0, 4].set_title(f"Mean sensitivity (nky={nky_less})")
    axes[0, 4].set_xlabel("horizontal direction (pixels)")
    axes[0, 4].set_ylabel("depth (pixels)")
    cbar4 = fig.colorbar(im4, ax=axes[0, 4], orientation='vertical', fraction=0.025, pad=0.02)
    cbar4.set_label("Sensitivity", rotation=270, labelpad=15)

    im5 = axes[1, 0].imshow(norm_log_resistivity_models.std(axis=0), cmap="viridis")
    axes[1, 0].set_title("Std Log Resistivity Model")
    axes[1, 0].set_xlabel("horizontal direction (pixels)")
    axes[1, 0].set_ylabel("depth (pixels)")
    cbar5 = fig.colorbar(im5, ax=axes[1, 0], orientation='vertical', fraction=0.025, pad=0.02)
    cbar5.set_label(r"Resistivity ($\Omega$m)", rotation=270, labelpad=15)

    im6 = axes[1, 1].imshow(std_accurate_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[1, 1].set_title(f"Std Pseudosection (nky={nky_accurate})")
    axes[1, 1].set_xlabel("horizontal direction")
    axes[1, 1].set_ylabel("pseudodepth")
    cbar6 = fig.colorbar(im6, ax=axes[1, 1], orientation='vertical', fraction=0.025, pad=0.02)
    cbar6.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=0)

    im7 = axes[1, 2].imshow(std_pseudosection, cmap="viridis", norm=norm_pseudo)
    axes[1, 2].set_title(f"Std Pseudosection (nky={nky_less})")
    axes[1, 2].set_xlabel("horizontal direction")
    axes[1, 2].set_ylabel("pseudodepth")
    cbar7 = fig.colorbar(im7, ax=axes[1, 2], orientation='vertical', fraction=0.025, pad=0.02)
    cbar7.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=0)

    im8 = axes[1, 3].imshow(std_accurate_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[1, 3].set_title(f"Std Sensitivity (nky={nky_accurate})")
    axes[1, 3].set_xlabel("horizontal direction (pixels)")
    axes[1, 3].set_ylabel("depth (pixels)")
    cbar8 = fig.colorbar(im8, ax=axes[1, 3], orientation='vertical', fraction=0.025, pad=0.02)
    cbar8.set_label("Sensitivity", rotation=270, labelpad=15)

    im9 = axes[1, 4].imshow(std_JtJ_diag, cmap="viridis", norm=norm_sensi)
    axes[1, 4].set_title(f"Std Sensitivity (nky={nky_less})")
    axes[1, 4].set_xlabel("horizontal direction (pixels)")
    axes[1, 4].set_ylabel("depth (pixels)")
    cbar9 = fig.colorbar(im9, ax=axes[1, 4], orientation='vertical', fraction=0.025, pad=0.02)
    cbar9.set_label("Sensitivity", rotation=270, labelpad=15)
    plt.suptitle(f"nky Test Results nky=[{nky_less}, {nky_accurate}] ({NUM_SAMPLES} samples)")
    plt.tight_layout()
    plt.savefig(save_dir / f"nky_test_means_std_{nky_less}_{nky_accurate}.png")
    plt.close(fig)

    mean_square_relative_difference_pseudosection = np.mean(np.square(accurate_pseudosections - pseudosections) / pseudosections, axis=0)
    mean_square_relative_difference_JtJ = np.mean(np.square(accurate_JtJ_diags - JtJ_diags) / JtJ_diags, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    vmin = np.nanmin(mean_square_relative_difference_pseudosection)
    vmax = np.nanmax(mean_square_relative_difference_pseudosection)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im0 = axes[0].imshow(mean_square_relative_difference_pseudosection, cmap="viridis", norm=norm)
    axes[0].set_title("Mean Square Relative Difference Pseudosection")
    axes[0].set_xlabel("horizontal direction")
    axes[0].set_ylabel("pseudodepth")
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.025, pad=0.02)
    cbar0.set_label(r"Apparent Resistivity ($\Omega$m)", rotation=270, labelpad=15)
    vmin = np.min(mean_square_relative_difference_JtJ)
    vmax = np.max(mean_square_relative_difference_JtJ)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im1 = axes[1].imshow(mean_square_relative_difference_JtJ, cmap="viridis", norm=norm)
    axes[1].set_title("Mean Square Relative Difference Sensitivity")
    axes[1].set_xlabel("horizontal direction (pixels)")
    axes[1].set_ylabel("depth (pixels)")
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.025, pad=0.02)
    cbar1.set_label("Sensitivity", rotation=270, labelpad=15)
    plt.suptitle(f"nky Test Mean Square Relative Difference nky=[{nky_less}, {nky_accurate}] ({NUM_SAMPLES} samples)")
    plt.tight_layout()
    plt.savefig(save_dir / f"nky_test_mean_square_relative_difference_{nky_less}_{nky_accurate}.png")
    plt.close(fig)
    

if __name__ == "__main__":
    dataset_path: Path = Path("/mnt/ensg/tout_le_monde/Basile/clean_reduced_unified")
    num_files = len(list(dataset_path.glob("*.npz")))
    SAVE_FOLDER: Path = Path("/mnt/ensg/tout_le_monde/Basile/forward_analysis")
    NUM_SAMPLES: int = 2048
    samples_per_npz = NUM_SAMPLES // 61 + 1

    air_test(NUM_SAMPLES, dataset_path, samples_per_npz, num_files, SAVE_FOLDER)
    # nky_test(NUM_SAMPLES, dataset_path, samples_per_npz, num_files, SAVE_FOLDER)
    # expMap_test(NUM_SAMPLES, dataset_path, samples_per_npz, num_files, SAVE_FOLDER)
    # inter_electrode_test(NUM_SAMPLES, dataset_path, samples_per_npz, num_files, SAVE_FOLDER)

