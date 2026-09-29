for a in somePath ;do directory=$(dirname $a);sct_deepseg spinalcord -i $a -qc ~/qc;done
