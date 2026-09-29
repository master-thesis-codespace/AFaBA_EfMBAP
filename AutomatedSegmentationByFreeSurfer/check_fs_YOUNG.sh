#!/usr/bin/env bash

# LIST="some files/paths ..."
# LIST="some files/paths ..."
LIST="some files/paths ..."

missing="some files/paths ..."
donefile="some files/paths ..."

> "$missing"
> "$donefile"

# WORKPATH=some files/paths ...
WORKPATH=.
while IFS= read -r T1
do
    # Get subject directory
    # SUBJ_DIR="$(dirname "$(dirname "$T1")")"
    SUBJ_DIR="$T1"

    # Expected FS output
    # some files/paths .../recon-all.done  
    OUTFILE="${WORKPATH}/${SUBJ_DIR}some files/paths ..."
    OUTFILE2="${WORKPATH}/${SUBJ_DIR}some files/paths ..."

    if [ -f "$OUTFILE" ] && [ -f "$OUTFILE2" ]; then
        echo "[OK] $OUTFILE"
        echo "$T1" >> "$donefile"
    else
        echo "[MISSING] $OUTFILE"
        echo "$T1" >> "$missing"
    fi

done < "$LIST"

echo "Finished checking."
echo "Completed subjects: $(wc -l < "$donefile")"
echo "Missing subjects:   $(wc -l < "$missing")"
