#!/usr/bin/env python3

import sys, os
import subprocess
from pathlib import Path

# CONFIG

SINGULARITY_IMAGE = "some files/paths ..."

LICENSE_FILE = "some files/paths ..."

WORK_DIR = "/directoryWhereTheMRIsAreStored"

# INPUT

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <T1_FILE>")
    sys.exit(1)

t1 = os.path.join(WORK_DIR, sys.argv[1])

if not os.path.exists(t1):
    raise FileNotFoundError(t1)

# SUBJECT DIRECTORY

# Example:
#
# some files/paths ...
#
# parents[1] = sub-XXX
#

subject_dir = os.path.dirname(t1)

# OUTPUT DIRECTORY

subjects_dir = os.path.join(subject_dir, "derivative")

os.makedirs(subjects_dir, exist_ok=True)

# FS SUBJECT ID

fsid = "FS"

# PRINT INFO

print("---")
print(f"T1: {t1}")
print(f"Subject Dir: {subject_dir}")
print(f"SUBJECTS_DIR: {subjects_dir}")
print(f"FS Folder: {os.path.join(subjects_dir, fsid)}")
print("---")

# COMMAND

cmd = [
    "singularity", "exec",
    "--cleanenv",

    "-B", f"{subject_dir}:{subject_dir}",
    "-B", f"{subjects_dir}:{subjects_dir}",
    "-B", f"{Path(LICENSE_FILE).parent}:{Path(LICENSE_FILE).parent}",

    SINGULARITY_IMAGE,

    "recon-all",
    "-sd", str(subjects_dir),
    "-s", fsid,
    "-i", str(t1),
    "-all"
]

# RUN

print("\nRunning command:\n")
print(" ".join(cmd))
print()

# subprocess.run(cmd, check=True)