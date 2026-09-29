for file in sub-*_T1w.nii.gz; do prefix="${file%%_*}"; mkdir -p "$prefix/anat" && mv "$file" "$prefix/anat/"; done
