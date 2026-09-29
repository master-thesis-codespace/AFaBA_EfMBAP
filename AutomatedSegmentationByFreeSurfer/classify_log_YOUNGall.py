#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

workpath = "."

INPUT_FILE = "some path/file ..."
OUTPUT_CSV = "some path/file ..."


def classify_log(log_text):
    txt = log_text.lower()
    issues = []

    # FOV issue
    if "cw256" in txt or ("fov=" in txt and "> 256" in txt):
        issues.append("FOV_GT_256")

    # Talairach
    tal_patterns = [
        "talairach alignment failed",
        "could not compute talairach transform",
        "talairach_avi",
    ]
    if any(p in txt for p in tal_patterns):
        issues.append("TALAIRACH")

    # Skull stripping / brainmask
    if (
        "error" in txt
        and (
            "brainmask" in txt
            or "mri_watershed" in txt
            or "skull strip" in txt
        )
    ):
        issues.append("SKULLSTRIP")

    # White matter segmentation
    if (
        "error" in txt
        and (
            "mri_fill" in txt
            or "fill failed" in txt
            or "white matter" in txt
        )
    ):
        issues.append("WM_SEG")

    # Memory
    mem_patterns = [
        "out of memory",
        "cannot allocate memory",
        "oom",
    ]
    if any(p in txt for p in mem_patterns):
        issues.append("OUT_OF_MEMORY")

    # Scheduler / Slurm kills
    kill_patterns = [
        "due to time limit",
        "cancelled at",
        "slurmstepd",
        "killed",
    ]
    if any(p in txt for p in kill_patterns):
        issues.append("TIME_LIMIT_OR_KILLED")

    # Generic recon-all failure
    if "exited with errors" in txt:
        issues.append("RECON_ALL_ERROR")

    # Successful completion
    if "finished without error" in txt:
        issues.append("LOG_SUCCESS")

    return issues


rows = []

with open(INPUT_FILE) as f:

    for line in f:

        subject_path = line.strip()

        if not subject_path:
            continue

        fs_dir = Path(workpath) / Path(subject_path) / "derivative" / "FS"

        log_file = fs_dir / "scripts" / "recon-all.log"

        row = {
            "subject_path": subject_path,
            "log_exists": False,
            "status": "",
            "issues": "",
            "aseg_mgz": False,
            "lh_pial": False,
            "rh_pial": False,
            "aseg_stats": False,
        }

        # Check outputs

        aseg = fs_dir / "mri" / "aseg.mgz"
        lh_pial = fs_dir / "surf" / "lh.pial"
        rh_pial = fs_dir / "surf" / "rh.pial"
        aseg_stats = fs_dir / "stats" / "aseg.stats"

        row["aseg_mgz"] = aseg.exists()
        row["lh_pial"] = lh_pial.exists()
        row["rh_pial"] = rh_pial.exists()
        row["aseg_stats"] = aseg_stats.exists()

        outputs_complete = all([
            row["aseg_mgz"],
            row["lh_pial"],
            row["rh_pial"],
            row["aseg_stats"],
        ])

        # Missing log

        if not log_file.exists():

            row["status"] = (
                "MISSING_LOG"
                if not outputs_complete
                else "OUTPUTS_PRESENT_NO_LOG"
            )

            rows.append(row)
            continue

        row["log_exists"] = True

        try:

            text = log_file.read_text(errors="ignore")

            issues = classify_log(text)

            row["issues"] = ";".join(issues)

            # Determine final status

            if outputs_complete:

                if "LOG_SUCCESS" in issues:
                    row["status"] = "SUCCESS"

                elif len(issues) == 0:
                    row["status"] = "OUTPUTS_COMPLETE"

                else:
                    row["status"] = "OUTPUTS_COMPLETE_WITH_WARNINGS"

            else:

                if "LOG_SUCCESS" in issues:
                    row["status"] = "INCOMPLETE_OUTPUTS"

                elif len(issues) > 0:
                    row["status"] = "FAILED"

                else:
                    row["status"] = "UNKNOWN_INCOMPLETE"

        except Exception as e:

            row["status"] = "READ_ERROR"
            row["issues"] = str(e)

        rows.append(row)

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")

print("\nStatus summary:")
print(df["status"].value_counts())
