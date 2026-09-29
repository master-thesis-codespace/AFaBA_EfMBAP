import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(".")
OUTPUT_CSV = "some path/file ..."

SCAN_TARGETS = {
    "exampleFolder1": [
        "dataset1",
        "dataset2",
    ],
    "exampleFolder2": [
        "dataset1",
        "dataset2",
    ],
}

def find_csv_for_subject(subject_dir: Path):                                                #Walk subject_dir at any depth to find a 'derivative' folder, then search recursively within it 
    for root, dirs, files in os.walk(subject_dir):
        if Path(root).name == "derivative":
            for d_root, _, d_files in os.walk(root):
                if "csa_perlevel.csv" in d_files:
                    return Path(d_root) / "csa_perlevel.csv"
            dirs[:] = [d for d in dirs if d != "derivative"]                                # Prevent outer walk from re-entering derivative subtree

    anat_csv = subject_dir / "some path/file ..." / "csa_perlevel.csv"                                    # Fallback: anat/ (Falls back to some path/file ... if not found)
    if anat_csv.is_file():
        return anat_csv

    return None

def iter_subjects(dataset_dir: Path, intermediate: str):                                    #gives (subject_name, subject_path) for every sub-* directory found directly inside dataset_dir / intermediate
    scan_dir = (dataset_dir / intermediate).resolve()
    if not scan_dir.is_dir():
        print(f"Attention: Intermediate path not found, skipping: {scan_dir}")
        return
    for entry in sorted(scan_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("sub-"):
            yield entry.name, entry

def main():
    all_frames = []
    missing_paths = []
    n_found = 0
    n_missing = 0

    for dataset_name, intermediates in SCAN_TARGETS.items():
        dataset_dir = BASE_DIR / dataset_name
        if not dataset_dir.is_dir():
            print(f"Attention: Dataset folder not found, skipping: {dataset_dir}")
            continue

        print(f"\nScanning {dataset_dir}...")

        for intermediate in intermediates:
            print(f"-> {intermediate}/")
            for subject_name, subject_path in iter_subjects(dataset_dir, intermediate):
                csv_path = find_csv_for_subject(subject_path)

                if csv_path is not None:
                    try:
                        df = pd.read_csv(csv_path)
                        df.insert(0, "Subject", subject_name)
                        df.insert(0, "Dataset", dataset_name)
                        all_frames.append(df)
                        n_found += 1
                    except Exception as exc:
                        print(f"Error: Could not read {csv_path}: {exc}")
                        missing_paths.append(str(csv_path) + f"[read error: {exc}]")
                        n_missing += 1
                else:
                    n_missing += 1
                    missing_paths.append(str(subject_path) + "/")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)                                     # Write output
        combined.to_csv(OUTPUT_CSV, index=False)
        print(f"\n Rerun CSV written -> '{OUTPUT_CSV}'")
        print(f"Rows: {len(combined):,}")
        print(f"Columns: {len(combined.columns)}")
        print(f"Subjects found: {n_found}")
        print(f"Subjects missing: {n_missing}")
    else:
        print("\n No data collected - output CSV was not written.")
        print(f"Subjects found: {n_found}")
        print(f"Subjects missing: {n_missing}")

    # Report missing
    if missing_paths:
        print(f"MISSING csa_perlevel.csv ({len(missing_paths)} subject(s)):")
        for p in missing_paths:
            print(f"{p}")
    else:
        print("\n No missing CSVs detected.")


if __name__ == "__main__":
    main()