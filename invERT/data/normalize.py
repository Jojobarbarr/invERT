import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import os # To get CPU count
import re
import cv2


def parse_stats_file(filepath="out.txt"):
    results = {}
    # Regex to capture the category name and the dictionary part separately
    # It looks for: Category Name: {content}
    line_pattern = re.compile(r"^\s*([^:]+):\s*(\{.*\})\s*$")
    # Regex to find the value associated with 'min': ...
    min_pattern = re.compile(r"'min':\s*([^,}]+)")
    # Regex to find the value associated with 'max': ... ending the dict
    max_pattern = re.compile(r"'max':\s*([^}]+)")
    # Regex to extract number from np.float32(...)
    np_float_pattern = re.compile(r"np\.float32\(([^)]+)\)")

    def convert_value(value_str):
        """Helper function to convert extracted string value to number."""
        value_str = value_str.strip()
        # Check for np.float32 format first
        np_match = np_float_pattern.search(value_str)
        if np_match:
            try:
                # Return as standard python float
                return float(np_match.group(1))
                # Or if you specifically need the numpy type:
                # return np.float32(np_match.group(1))
            except ValueError:
                print(f"Warning: Could not convert numpy float value: {value_str}")
                return None
        else:
            # Try converting to int, then float
            try:
                return int(value_str)
            except ValueError:
                try:
                    return float(value_str)
                except ValueError:
                    print(f"Warning: Could not convert value: {value_str}")
                    return None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = line_pattern.match(line)
                if match:
                    category_name = match.group(1).strip()
                    dict_content = match.group(2)

                    min_match = min_pattern.search(dict_content)
                    max_match = max_pattern.search(dict_content)

                    if min_match and max_match:
                        min_val_str = min_match.group(1)
                        max_val_str = max_match.group(1)

                        min_val = convert_value(min_val_str)
                        max_val = convert_value(max_val_str)

                        if min_val is not None and max_val is not None:
                            results[category_name] = {'min': min_val, 'max': max_val}
                        else:
                             print(f"Warning: Failed converting values for category '{category_name}'")
                    else:
                        print(f"Warning: Could not parse min/max from dict content in line: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

    return results


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
        np.clip(JtJ_diag, 0.05, 1, out=JtJ_diag)

        pseudosection = cv2.resize(pseudosection, (93, 49), interpolation=cv2.INTER_LINEAR)
        norm_log_resistivity_model = cv2.resize(norm_log_resistivity_model, (256, 192), interpolation=cv2.INTER_LINEAR)
        JtJ_diag = cv2.resize(JtJ_diag, (256, 192), interpolation=cv2.INTER_LINEAR)
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
    in_path = Path("/mnt/ensg/tout_le_monde/Basile/dataset_sensitivity_5_5")

    out_path = in_path / "samples_normalized_ps_resized"
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

    if (in_path / "overall_statistics.txt").exists():
        print("Overall statistics file found. Loading existing statistics...")
        with open(in_path / "overall_statistics.txt", 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("Num Electrode:"):
                    # Extract values using eval safely
                    values = eval(line.split(":", 1)[1].strip())
                    overall_stat_num_electrode["min"] = values["min"]
                    overall_stat_num_electrode["max"] = values["max"]

                elif line.startswith("Subsection Length:"):
                    values = eval(line.split(":", 1)[1].strip())
                    overall_stat_subsection_length["min"] = values["min"]
                    overall_stat_subsection_length["max"] = values["max"]

                elif line.startswith("Log1p Pseudosection:"):
                    values_str = line.split(":", 1)[1].strip()
                    # Replace np.float32(...) with actual float values
                    values_str = values_str.replace("np.float32", "")
                    values = eval(values_str)
                    overall_stat_pseudosection["min"] = float(values["min"])
                    overall_stat_pseudosection["max"] = float(values["max"])
    else:
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

    # Save overall statistics to a file
    stats_file = out_path.parent / "overall_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("Overall Statistics:\n")
        f.write(f"Num Electrode: {overall_stat_num_electrode}\n")
        f.write(f"Subsection Length: {overall_stat_subsection_length}\n")
        f.write(f"Log1p Pseudosection: {overall_stat_pseudosection}\n")
    print(f"Overall statistics saved to {stats_file}")

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

