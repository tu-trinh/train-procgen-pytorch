#!/bin/bash

env_name=$1
detector_type=$2
directories=()
for f in ./logs/procgen/${env_name}/*_svdd_${detector_type}*; do
    if [ -d "$f" ]; then
        for subdir in "$f"/*; do
            if [ -d "$subdir" ]; then
                directories+=("$subdir")
            fi
        done
    fi
done

for i in "${!directories[@]}"; do
    python3 animate.py -d ${directories[i]}
done
