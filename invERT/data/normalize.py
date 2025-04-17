import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import os # To get CPU count


def process_file(file_path):
    """
    Loads a single npz file, extracts required data, performs necessary
    calculations, and returns the relevant values for aggregation.
    """
    try:
        sample = np.load(file_path)

        # Load only necessary data
        num_electrode = sample['num_electrode'].item()
        subsection_length = sample['subsection_length'].item()
        pseudosection = np.log1p(sample['pseudosection'])

        # Calculate min/max for this file's pseudosection
        pseudo_min = np.min(pseudosection)
        pseudo_max = np.max(pseudosection)

        # Return values needed for overall statistics
        return (num_electrode, subsection_length, pseudo_min, pseudo_max)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        # Return neutral values that won't affect min/max calculation
        # Or handle error differently (e.g., return None and filter later)
        return (np.inf, np.inf, np.inf, -np.inf)


def save_new_npz(in_file_path, out_file_path, overall_stat_num_electrode, overall_stat_subsection_length, overall_stat_pseudosection):
    try:
        sample = np.load(in_file_path)

        # Load only necessary data
        num_electrode = sample['num_electrode'].item()
        subsection_length = sample['subsection_length'].item()
        array_type = sample['array_type']
        pseudosection = np.log1p(sample['pseudosection'])
        norm_log_resistivity_model = sample['norm_log_resistivity_model']
        JtJ_diag = np.log(sample['JtJ_diag'])

        min_JtJ_diag = np.min(JtJ_diag)
        max_JtJ_diag = np.max(JtJ_diag)

        num_electrode = (num_electrode - overall_stat_num_electrode["min"]) / (overall_stat_num_electrode["max"] - overall_stat_num_electrode["min"])
        subsection_length = (subsection_length - overall_stat_subsection_length["min"]) / (overall_stat_subsection_length["max"] - overall_stat_subsection_length["min"])
        pseudosection = (pseudosection - overall_stat_pseudosection["min"]) / (overall_stat_pseudosection["max"] - overall_stat_pseudosection["min"])
        JtJ_diag = (JtJ_diag - min_JtJ_diag) / (max_JtJ_diag - min_JtJ_diag)
        np.clip(JtJ_diag, 0.1, 1, out=JtJ_diag)

        sample_out = {
            'num_electrode': num_electrode,
            'subsection_length': subsection_length,
            'array_type': array_type,
            'pseudosection': pseudosection,
            'norm_log_resistivity_model': norm_log_resistivity_model,
            'JtJ_diag': JtJ_diag,
        }

        # Save the new npz file
        np.savez_compressed(out_file_path, **sample_out)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


if __name__ == "__main__":
    in_path = Path("/mnt/ensg/tout_le_monde/Basile/dataset_sensitivity_3_3")

    out_path = in_path / "samples_normalized"
    out_path.mkdir(parents=True, exist_ok=True)

    samples_path = in_path / "samples"

    # Get files without sorting (usually faster if order doesn't matter)
    # Convert generator to list to get the total count for tqdm
    files = list(samples_path.glob("*.npz"))
    if not files:
        print("No .npz files found in the directory.")
        exit()

    # --- Initialize Overall Statistics ---
    overall_stat_num_electrode = {
        "min": np.inf,
        "max": -np.inf,
    }
    overall_stat_subsection_length = {
        "min": np.inf,
        "max": -np.inf,
    }
    overall_stat_pseudosection = {
        "min": np.inf,
        "max": -np.inf,
    }

    # --- Parallel Processing ---
    # Adjust max_workers based on your system's cores and I/O limits
    # None usually defaults to the number of processors on the machine
    num_workers = os.cpu_count() * 16
    results = []

    print(f"Processing {len(files)} files using up to {num_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_file, file): file for file in files}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files), desc="Computing min/max samples"):
           try:
               result = future.result()
               results.append(result)
           except Exception as exc:
                file_path = future_to_file[future]
                print(f'{file_path} generated an exception: {exc}')


    # --- Aggregate Results ---
    print("Aggregating results...")
    for ne, sl, p_min, p_max in results:
        # Skip potential error results if they were marked (e.g., with None)
        if ne is None: # Adjust if using a different error indicator
            continue

        overall_stat_num_electrode["min"] = min(overall_stat_num_electrode["min"], ne)
        overall_stat_num_electrode["max"] = max(overall_stat_num_electrode["max"], ne)
        overall_stat_subsection_length["min"] = min(overall_stat_subsection_length["min"], sl)
        overall_stat_subsection_length["max"] = max(overall_stat_subsection_length["max"], sl)
        overall_stat_pseudosection["min"] = min(overall_stat_pseudosection["min"], p_min)
        overall_stat_pseudosection["max"] = max(overall_stat_pseudosection["max"], p_max)

    # --- Print Final Statistics ---
    print("\n--- Final Statistics ---")
    print(f"Num Electrode: {overall_stat_num_electrode}")
    print(f"Subsection Length: {overall_stat_subsection_length}")
    print(f"Log1p Pseudosection: {overall_stat_pseudosection}")

    # --- Normalize and Save ---
    print("\nNormalizing and saving files...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(save_new_npz, file, out_path / file.name, overall_stat_num_electrode, overall_stat_subsection_length, overall_stat_pseudosection): file for file in files}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files), desc="Saving normalized samples"):
            try:
                future.result()
            except Exception as exc:
                file_path = future_to_file[future]
                print(f'{file_path} generated an exception: {exc}')
    
    print("All files processed and saved successfully.")

