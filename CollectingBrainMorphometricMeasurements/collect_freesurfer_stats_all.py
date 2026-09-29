import os
import csv
import pandas as pd
from pathlib import Path
from collections import OrderedDict

BASE_DIR = Path(".")
OUTPUT_CSV = "some path/file ..."

SCAN_TARGETS = {
    "datasetFolder1": ["."],
    "datasetFolder2": ["."],
    "datasetFolder3": ["."],
}

STATS_FILES = ["aseg.stats", "lh.aparc.a2009s.stats", "rh.aparc.a2009s.stats"]
PREFIXES = ["aseg", "lh_a2009s", "rh_a2009s"]

SPECIFIC_REGIONS = [
    'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
    'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen',
    'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus',
    'Left-Amygdala', 'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel',
    'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
    'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex', 'Right-Thalamus-Proper',
    'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
    'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel', 'Right-choroid-plexus',
    'Optic-Chiasm', 'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior',
    'CC_Anterior'
]

REGION_ALIASES = {                                                                                  # FreeSurfer 7 dropped "-Proper" from thalamus - map here
    'Left-Thalamus': 'Left-Thalamus-Proper',
    'Right-Thalamus': 'Right-Thalamus-Proper',
}

GLOBAL_MEASURES = [
    'BrainSegVol', 'BrainSegVolNotVent', 'BrainSegVolNotVentSurf', 'VentricleChoroidVol',
    'lhCortexVol', 'rhCortexVol', 'CortexVol', 'lhCerebralWhiteMatterVol',
    'rhCerebralWhiteMatterVol', 'CerebralWhiteMatterVol', 'SubCortGrayVol', 'TotalGrayVol',
    'SupraTentorialVol', 'SupraTentorialVolNotVent', 'SupraTentorialVolNotVentVox',
    'MaskVol', 'BrainSegVol-to-eTIV', 'MaskVol-to-eTIV', 'lhSurfaceHoles',
    'rhSurfaceHoles', 'SurfaceHoles', 'eTIV'
]

APARC_ATTRS = ['SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd']

def parse_aseg_stats(filepath: Path):                                                               #Extract from aseg.stats global measures (volume) and normMean and normStdDev for each structure
    result = OrderedDict()
    global_set = set(GLOBAL_MEASURES)
    specific_set = set(SPECIFIC_REGIONS)

    with open(filepath) as fh:                                                                      # global volumes from # Measure lines
        for line in fh:
            if not line.startswith('# Measure'):
                continue
            parts = [p.strip() for p in line.split(',')]
            # Measure <tag>, <ShortName>, <LongName>, <value>, <unit>
            if len(parts) < 4:
                continue
            short_name = parts[1].strip()
            if short_name in global_set:
                try:
                    val = float(parts[3].strip())
                except ValueError:
                    val = parts[3].strip()
                result[f"aseg_{short_name}_Vol"] = val

    headers = None                                                                                  #regional normMean / normStdDev
    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('# ColHeaders'):
                headers = line.replace('# ColHeaders', '').split()
                continue
            if line.startswith('#') or not line.strip():
                continue
            if headers is None:
                continue
            row = line.split()
            if len(row) < len(headers):
                continue
            struct_col = headers.index('StructName')
            raw_name = row[struct_col]
            canonical = REGION_ALIASES.get(raw_name, raw_name)
            if canonical not in specific_set:
                continue
            for attr in ('normMean', 'normStdDev'):
                if attr in headers:
                    val = row[headers.index(attr)]
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    result[f"aseg_{canonical}_{attr}"] = val

    return result


def parse_aparc_a2009s_stats(filepath: Path, prefix: str, is_lh: bool, aparc_global_seen: set):     #extract from lh/rh.aparc.a2009s.stats regional SurfArea, GrayVol, ThickAvg, ThickStd and global measures from '# Measure' if not already covered
    result = OrderedDict()
    gm_set = set(GLOBAL_MEASURES)

    if is_lh:                                                                                       # Global measures from comment lines without duplication
        with open(filepath) as fh:
            for line in fh:
                if not line.startswith('# Measure'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 4:
                    continue
                short_name = parts[1].strip()
                if short_name in gm_set and short_name not in aparc_global_seen:
                    col = f"aseg_{short_name}_Vol"
                    try:
                        val = float(parts[3].strip())
                    except ValueError:
                        val = parts[3].strip()
                    result[col] = val
                    aparc_global_seen.add(short_name)

    headers = None                                                                                  #Regional attributes
    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('# ColHeaders'):
                headers = line.replace('# ColHeaders', '').split()
                continue
            if line.startswith('#') or not line.strip():
                continue
            if headers is None:
                continue
            row = line.split()
            if len(row) < len(headers):
                continue
            struct_col= headers.index('StructName')
            struct_name = row[struct_col]
            for attr in APARC_ATTRS:
                if attr in headers:
                    val = row[headers.index(attr)]
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    result[f"{prefix}_{struct_name}_{attr}"] = val
    return result


def collect_subject_row(dataset_name: str, subject_name: str, stats_dir: Path):                     #one OrderedDict for a subject (Dataset and Subject always first two columns)
    aparc_global_seen = set()                                                                       # track globals already captured from aseg

    row = OrderedDict()
    row["Dataset"] = dataset_name
    row["Subject"] = subject_name

    aseg_data = parse_aseg_stats(stats_dir / "aseg.stats")                                          # aseg - globals + regional normMean/normStdDev
    for key in aseg_data:                                                                           #Mark which globals were already captured from aseg
        if key.startswith("aseg_") and key.endswith("_Vol"):
            short = key[len("aseg_"):-len("_Vol")]
            aparc_global_seen.add(short)
    row.update(aseg_data)

    row.update(parse_aparc_a2009s_stats(                                                            # lh - regional att + any globals not yet seen
        stats_dir / "lh.aparc.a2009s.stats",
        "lh_a2009s", is_lh=True, aparc_global_seen=aparc_global_seen
    ))

    row.update(parse_aparc_a2009s_stats(                                                            #rh- regional attrs only (no globals)
        stats_dir / "rh.aparc.a2009s.stats",
        "rh_a2009s", is_lh=False, aparc_global_seen=aparc_global_seen
    ))

    return row


def find_stats_dir(subject_dir: Path):                                                              #Recursively search for a 'derivative' folder inside subject_dir, returns if found
    try:
        for root, dirs, _ in os.walk(subject_dir, onerror=lambda e: None):
            if Path(root).name == "derivative":
                stats_dir = Path(root) / "FS" / "stats"
                try:
                    if stats_dir.is_dir():
                        return stats_dir
                except PermissionError:
                    pass
                dirs[:] = []                                                                            # do not descend further once derivative found
    except PermissionError:
        pass
    return None

def iter_subjects(dataset_dir: Path, intermediate: str):                                                #gives (subject_name, subject_path) for sub-* dirs inside dataset_dir/intermediate
    scan_dir = (dataset_dir / intermediate).resolve()
    if not scan_dir.is_dir():
        print(f"Achtung: Intermediate path not found, skipping: {scan_dir}")
        return
    for entry in sorted(scan_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("sub-"):
            yield entry.name, entry

def main():
    all_rows = []
    missing_paths = []
    n_found = 0
    n_missing = 0

    for dataset_key, intermediates in SCAN_TARGETS.items():
        dataset_name = Path(dataset_key).name
        dataset_dir  = BASE_DIR / dataset_key

        if not dataset_dir.is_dir():
            print(f"Achtung: Dataset folder not found, skipping: {dataset_dir}")
            continue

        print(f"\nScanning {dataset_dir} ...")

        for intermediate in intermediates:
            print(f"-> {intermediate}/")

            for subject_name, subject_path in iter_subjects(dataset_dir, intermediate):
                stats_dir = find_stats_dir(subject_path)

                subject_missing = []                                                                    # Determine which files are missing
                if stats_dir is None:
                    subject_missing = STATS_FILES[:]
                else:
                    for fname in STATS_FILES:
                        if not (stats_dir / fname).is_file():
                            subject_missing.append(fname)

                if subject_missing:
                    n_missing += 1
                    missing_paths.append(f"{subject_path}  [missing: {', '.join(subject_missing)}]")
                    continue

                try:                                                                                        # Parse all three files into one row
                    row = collect_subject_row(dataset_name, subject_name, stats_dir)
                    all_rows.append(row)
                    n_found += 1

                except Exception as exc:
                    n_missing += 1
                    missing_paths.append(
                        f"{subject_path}  [parse error: {exc}]"
                    )

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nWritten -> '{OUTPUT_CSV}'")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns):,}")
        print(f"Subjects OK: {n_found:,}")
        print(f"Subjects missing: {n_missing:,}")
    else:
        print("\nNo data collected - output CSV was not written.")
        print(f"Subjects found: {n_found}")
        print(f"Subjects missing: {n_missing}")

    if missing_paths:
        print(f"SUBJECTS WITH MISSING FILES  ({len(missing_paths)} subject(s)):")
        for p in missing_paths:
            print(f"{p}")
    else:
        print("\nNo missing stats files detected.")

if __name__ == "__main__":
    main()