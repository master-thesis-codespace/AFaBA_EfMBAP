# coding: utf-8
import os
import re

import pandas as pd
import pytest

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "some path/file ...")

EXPECTED_COLUMNS = [
    "Dataset", "Subject", "SCT Version", "Filename", "Slice (I->S)", "VertLevel", "DistancePMJ",
    "MEAN(area)", "STD(area)", "MEAN(angle_AP)", "STD(angle_AP)", "MEAN(angle_RL)", "STD(angle_RL)",
    "MEAN(diameter_AP)", "STD(diameter_AP)", "MEAN(diameter_RL)", "STD(diameter_RL)",
    "MEAN(eccentricity)", "STD(eccentricity)", "MEAN(orientation)", "STD(orientation)",
    "MEAN(solidity)", "STD(solidity)", "SUM(length)",
]

MEASURE_COLUMNS = [                                                                                     # Numeric measurement columns (everything except the metadata/string columns and VertLevel)
    "MEAN(area)", "STD(area)", "MEAN(angle_AP)", "STD(angle_AP)", "MEAN(angle_RL)", "STD(angle_RL)",
    "MEAN(diameter_AP)", "STD(diameter_AP)", "MEAN(diameter_RL)", "STD(diameter_RL)",
    "MEAN(eccentricity)", "STD(eccentricity)", "MEAN(orientation)", "STD(orientation)",
    "MEAN(solidity)", "STD(solidity)", "SUM(length)",
]

STD_COLUMNS = [c for c in MEASURE_COLUMNS if c.startswith("STD(")]

STRING_COLUMNS = ["Dataset", "Subject", "SCT Version", "Filename", "Slice (I->S)"]

COLUMNS_ALLOWED_NAN = {"DistancePMJ"}                                                                   # Columns allowed to contain NaN. DistancePMJ is a known, always-empty column in this pipeline

VALID_VERTLEVEL_SETS = [frozenset({1, 2, 3}), frozenset({1, 2, 3, 4})]

@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV_PATH)

def test_file_exists():
    assert os.path.exists(CSV_PATH), f"File not found: {CSV_PATH}"

def test_file_loads_without_error():
    loaded = pd.read_csv(CSV_PATH)
    assert loaded is not None

def test_column_count(df):
    assert len(df.columns) == len(EXPECTED_COLUMNS), (f"Expected {len(EXPECTED_COLUMNS)} columns, found {len(df.columns)}: {list(df.columns)}")


def test_column_names_and_order(df):
    assert list(df.columns) == EXPECTED_COLUMNS, (f"Column names/order mismatch.\nExpected: {EXPECTED_COLUMNS}\nActual: {list(df.columns)}")

def test_no_duplicate_column_names(df):
    assert len(df.columns) == len(set(df.columns)), "Duplicate column names found"

def test_has_at_least_one_row(df):
    assert len(df) > 0, "CSV has no data rows"

def test_vertlevel_is_integer(df):
    assert pd.api.types.is_integer_dtype(df["VertLevel"]), (f"VertLevel is not integer-typed: {df['VertLevel'].dtype}")

@pytest.mark.parametrize("col", MEASURE_COLUMNS)
def test_measure_columns_are_numeric(df, col):
    assert pd.api.types.is_numeric_dtype(df[col]), (f"Column '{col}' is not numeric (dtype={df[col].dtype}) - a non-numeric value (e.g. 'NA', '#DIV/0!')")

def test_slice_format(df):
    pattern = re.compile(r"^\d+:\d+$")
    bad = df.loc[~df["Slice (I->S)"].astype(str).str.match(pattern), "Slice (I->S)"]
    assert bad.empty, f"Rows with unexpected 'Slice (I->S)' format (expected '<int>:<int>'): {bad.tolist()}"

@pytest.mark.parametrize("col", [c for c in EXPECTED_COLUMNS if c not in COLUMNS_ALLOWED_NAN])
def test_no_missing_values_except_allowlisted(df, col):
    n_missing = df[col].isna().sum()
    assert n_missing == 0, f"Column '{col}' has {n_missing} missing value(s) - not in the allowed-NaN list"

def test_distancepmj_is_still_fully_empty(df):
    n_present = df["DistancePMJ"].notna().sum()
    assert n_present == 0, (f"DistancePMJ now has {n_present} non-null value(s) - previously always fully empty. ")

@pytest.mark.parametrize("col", STRING_COLUMNS)
def test_no_empty_string_cells(df, col):
    empty = df[col].astype(str).str.strip() == ""
    assert not empty.any(), f"Column '{col}' has {empty.sum()} empty-string cell(s)"

def test_no_fully_empty_rows(df):
    empty_rows = df.index[df.isna().all(axis=1)].tolist()
    assert not empty_rows, f"Fully empty row(s) found at index: {empty_rows}"

def test_no_unexpected_fully_empty_columns(df):
    empty_cols = set(df.columns[df.isna().all()]) - COLUMNS_ALLOWED_NAN
    assert not empty_cols, f"Unexpected fully-empty column(s): {sorted(empty_cols)}"

def test_area_positive(df):
    assert (df["MEAN(area)"] > 0).all(), "MEAN(area) has non-positive value(s)"
    assert (df["STD(area)"] >= 0).all(), "STD(area) has negative value(s)"

@pytest.mark.parametrize("col", STD_COLUMNS)
def test_std_columns_non_negative(df, col):
    n_neg = (df[col] < 0).sum()
    assert n_neg == 0, f"Column '{col}' has {n_neg} negative value(s) - a standard deviation cannot be negative"

def test_diameters_positive(df):
    assert (df["MEAN(diameter_AP)"] > 0).all(), "MEAN(diameter_AP) has non-positive value(s)"
    assert (df["MEAN(diameter_RL)"] > 0).all(), "MEAN(diameter_RL) has non-positive value(s)"

def test_solidity_in_valid_range(df):
    tol = 1e-6
    assert df["MEAN(solidity)"].between(0 - tol, 1 + tol).all(), (
        f"MEAN(solidity) out of expected (0,1] range: "
        f"min={df['MEAN(solidity)'].min()}, max={df['MEAN(solidity)'].max()}"
    )

def test_eccentricity_in_valid_range(df):
    tol = 1e-6
    assert df["MEAN(eccentricity)"].between(0 - tol, 1 + tol).all(), (
        f"MEAN(eccentricity) out of expected [0,1] range: "
        f"min={df['MEAN(eccentricity)'].min()}, max={df['MEAN(eccentricity)'].max()}"
    )

def test_sum_length_positive(df):
    assert (df["SUM(length)"] > 0).all(), "SUM(length) has non-positive value(s)"

def test_no_duplicate_subject_vertlevel_pairs(df):
    dup = df.duplicated(subset=["Subject", "VertLevel"], keep=False)
    assert not dup.any(), (
        f"Duplicate (Subject, VertLevel) pairs found:\n"
        f"{df.loc[dup, ['Subject', 'VertLevel']].sort_values(['Subject', 'VertLevel'])}"
    )

def test_vertlevel_set_per_subject_is_valid(df):
    invalid = {}
    for subject, group in df.groupby("Subject"):
        levels = group["VertLevel"].tolist()
        level_set = frozenset(levels)
        if len(levels) != len(level_set) or level_set not in VALID_VERTLEVEL_SETS:                                      # catches duplicates within a subject (set collapses them, so compare lengths too)
            invalid[subject] = sorted(levels)
    assert not invalid, (f"Subject(s) with invalid VertLevel sets (must be exactly {{1,2,3}} or {{1,2,3,4}}, no duplicates/gaps): {invalid}")

def test_subject_naming_convention(df):
    bad = df.loc[~df["Subject"].astype(str).str.match(r"^sub-"), "Subject"].unique()
    assert len(bad) == 0, f"Subject value(s) not matching 'sub-' prefix convention: {list(bad)}"

def test_dataset_values_are_known(df):
    KNOWN_DATASETS = {"some files/paths ...", "some files/paths ..."}
    unknown = set(df["Dataset"].unique()) - KNOWN_DATASETS
    assert not unknown, f"Unexpected Dataset value(s) not in the known allowlist: {unknown}"

def test_every_row_has_expected_field_count():
    raw = pd.read_csv(CSV_PATH, dtype=str)
    assert raw.shape[1] == len(EXPECTED_COLUMNS), (
        f"Row field count mismatch when reading as raw strings: {raw.shape[1]} columns (expected {len(EXPECTED_COLUMNS)})"
    )
    assert not any(col.startswith("Unnamed:") for col in raw.columns), (
        f"Found 'Unnamed:' column(s), indicating a header/row misalignment: {list(raw.columns)}"
    )

def test_no_fully_duplicate_rows(df):
    dup = df.duplicated(keep=False)
    assert not dup.any(), f"{dup.sum()} fully duplicate row(s) found at index: {df.index[dup].tolist()}"

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))