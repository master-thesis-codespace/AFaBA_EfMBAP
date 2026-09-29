for a in somePath ;do directory=$(dirname $a); sct_label_utils -i $a -create-viewer 3 -o ${directory}/label_c2c3.nii.gz;done
