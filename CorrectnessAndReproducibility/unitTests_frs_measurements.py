# coding: utf-8
import os
import re

import pandas as pd
import pytest

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "some path/file ...")

METADATA_COLUMNS = ["Dataset", "Subject"]

SUFFIX_RULES = {                                                                                        # suffix regex -> allowed prefix families for that suffix
    "_Vol": ("aseg_",),
    "_normMean": ("aseg_",),
    "_normStdDev": ("aseg_",),
    "_SurfArea": ("lh_a2009s_", "rh_a2009s_"),
    "_GrayVol": ("lh_a2009s_", "rh_a2009s_"),
    "_ThickAvg": ("lh_a2009s_", "rh_a2009s_"),
    "_ThickStd": ("lh_a2009s_", "rh_a2009s_"),
}

SUFFIX_ORDER = sorted(SUFFIX_RULES, key=len, reverse=True)

COLUMN_MISSINGNESS_THRESHOLD = 0.05                                                                     # 5%
NORMMEAN_UPPER_BOUND = 1000                                                                             # loose outlier/unit-error guard
THICKAVG_UPPER_BOUND = 8.0                                                                              # loose sanity bound in mm

@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV_PATH)

@pytest.fixture(scope="module")
def feature_columns(df):
    return [c for c in df.columns if c not in METADATA_COLUMNS]

def _suffix_of(col: str):
    for suffix in SUFFIX_ORDER:
        if col.endswith(suffix):
            return suffix
    return None

def _columns_with_suffix(columns, suffix):
    return [c for c in columns if c.endswith(suffix)]

def test_file_exists():
    assert os.path.exists(CSV_PATH), f"File not found: {CSV_PATH}"

def test_file_loads_without_error():
    loaded = pd.read_csv(CSV_PATH)
    assert loaded is not None

def test_first_two_columns_are_metadata(df):
    assert list(df.columns[:2]) == METADATA_COLUMNS, (f"Expected first two columns to be {METADATA_COLUMNS}, found {list(df.columns[:2])}")

def test_all_feature_columns_match_known_naming_pattern(feature_columns):
    unmatched_suffix = []
    wrong_prefix = []
    for col in feature_columns:
        suffix = _suffix_of(col)
        if suffix is None:
            unmatched_suffix.append(col)
            continue
        allowed_prefixes = SUFFIX_RULES[suffix]
        if not col.startswith(allowed_prefixes):
            wrong_prefix.append((col, suffix, allowed_prefixes))

    assert not unmatched_suffix, (
        f"{len(unmatched_suffix)} column(s) do not end in any known suffix "
        f"{list(SUFFIX_RULES)}: {unmatched_suffix[:20]}{' ...' if len(unmatched_suffix) > 20 else ''}"
    )
    assert not wrong_prefix, (
        f"{len(wrong_prefix)} column(s) have the right suffix but an unexpected prefix: "
        f"{wrong_prefix[:20]}{' ...' if len(wrong_prefix) > 20 else ''}"
    )

def test_no_duplicate_column_names(df):
    assert len(df.columns) == len(set(df.columns)), "Duplicate column names found"

def test_has_at_least_one_row(df):
    assert len(df) > 0, "CSV has no data rows"

def test_feature_columns_are_numeric(df, feature_columns):
    non_numeric = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(df[c])]
    assert not non_numeric, (
        f"{len(non_numeric)} feature column(s) are not numeric - a non-numeric value may have leaked in: {non_numeric[:20]}{' ...' if len(non_numeric) > 20 else ''}"
    )

def test_metadata_columns_are_string_dtype(df):
    for col in METADATA_COLUMNS:
        assert pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]), (
            f"Column '{col}' expected to be string-typed, found dtype={df[col].dtype}"
        )

@pytest.mark.parametrize("col", METADATA_COLUMNS)
def test_metadata_columns_have_no_missing_values(df, col):
    n_missing = df[col].isna().sum()
    assert n_missing == 0, f"Metadata column '{col}' has {n_missing} missing value(s) - must always be complete"


def test_no_column_exceeds_missingness_threshold(df, feature_columns):
    na_frac = df[feature_columns].isna().mean()
    violations = na_frac[na_frac > COLUMN_MISSINGNESS_THRESHOLD]
    assert violations.empty, (
        f"{len(violations)} column(s) exceed the {COLUMN_MISSINGNESS_THRESHOLD:.0%} missingness "
        f"ceiling (possible systemic gap, not just scattered per-subject noise):\n{violations}"
    )

def test_no_subject_missing_all_features(df, feature_columns):
    fully_missing = df.index[df[feature_columns].isna().all(axis=1)]
    assert len(fully_missing) == 0, (f"{len(fully_missing)} subject(s) missing ALL {len(feature_columns)} feature values: {df.loc[fully_missing, 'Subject'].tolist()}")

def test_no_fully_empty_columns(df):
    empty_cols = df.columns[df.isna().all()].tolist()
    assert not empty_cols, f"Fully-empty column(s) found: {empty_cols}"

@pytest.mark.parametrize("col", METADATA_COLUMNS)
def test_no_empty_string_cells_in_metadata(df, col):
    empty = df[col].astype(str).str.strip() == ""
    assert not empty.any(), f"Column '{col}' has {empty.sum()} empty-string cell(s)"

def test_vol_grayvol_surfarea_non_negative(df, feature_columns):
    cols = (_columns_with_suffix(feature_columns, "_Vol") + _columns_with_suffix(feature_columns, "_GrayVol") + _columns_with_suffix(feature_columns, "_SurfArea"))
    assert cols, "No Vol/GrayVol/SurfArea columns found - suffix matching may be broken"
    sub = df[cols]
    n_negative = (sub < 0).sum()
    violations = n_negative[n_negative > 0]
    assert violations.empty, f"Negative value(s) found in volume/area column(s) (physically impossible):\n{violations}"

def test_thickavg_within_plausible_range(df, feature_columns):
    cols = _columns_with_suffix(feature_columns, "_ThickAvg")
    assert cols, "No ThickAvg columns found - suffix matching may be broken"
    sub = df[cols]
    too_low = (sub <= 0).sum()
    too_high = (sub > THICKAVG_UPPER_BOUND).sum()
    bad = (too_low + too_high)
    violations = bad[bad > 0]
    assert violations.empty, (f"ThickAvg value(s) outside plausible (0, {THICKAVG_UPPER_BOUND}]mm range:\n{violations}")


def test_thickstd_normstddev_non_negative(df, feature_columns):
    cols = (_columns_with_suffix(feature_columns, "_ThickStd") + _columns_with_suffix(feature_columns, "_normStdDev"))
    assert cols, "No ThickStd/normStdDev columns found - suffix matching may be broken"
    sub = df[cols]
    n_negative = (sub < 0).sum()
    violations = n_negative[n_negative > 0]
    assert violations.empty, f"Negative value(s) found in a standard-deviation column (impossible):\n{violations}"

def test_normmean_within_plausible_range(df, feature_columns):
    cols = _columns_with_suffix(feature_columns, "_normMean")
    assert cols, "No normMean columns found - suffix matching may be broken"
    sub = df[cols]
    too_low = (sub < 0).sum()
    too_high = (sub > NORMMEAN_UPPER_BOUND).sum()
    bad = (too_low + too_high)
    violations = bad[bad > 0]
    assert violations.empty, (f"normMean value(s) outside plausible [0, {NORMMEAN_UPPER_BOUND}) range:\n{violations}")

def test_no_duplicate_subjects(df):
    dup = df["Subject"].duplicated(keep=False)
    assert not dup.any(), f"Duplicate Subject value(s) found: {df.loc[dup, 'Subject'].unique().tolist()}"

def test_subject_naming_convention(df):
    bad = df.loc[~df["Subject"].astype(str).str.match(r"^sub-"), "Subject"].tolist()
    assert not bad, f"Subject value(s) not matching 'sub-' prefix convention: {bad[:20]}"

def test_dataset_is_non_empty_string(df):
    bad = df["Dataset"].isna() | (df["Dataset"].astype(str).str.strip() == "")
    assert not bad.any(), f"{bad.sum()} row(s) with a missing/empty Dataset value"

def test_every_row_has_expected_field_count(df):
    raw = pd.read_csv(CSV_PATH, dtype=str)
    assert raw.shape[1] == len(df.columns), (f"Row field count mismatch when reading as raw strings: {raw.shape[1]} columns (expected {len(df.columns)}) - possible ragged row from an unescaped delimiter")
    assert not any(col.startswith("Unnamed:") for col in raw.columns), (f"Found 'Unnamed:' column(s), indicating a header/row misalignment")

def test_no_fully_duplicate_rows(df):
    dup = df.duplicated(keep=False)
    assert not dup.any(), f"{dup.sum()} fully duplicate row(s) found at index: {df.index[dup].tolist()}"

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))