#!/bin/bash

directories=()
for f in ./logs/procgen/coinrun_aisc/receive_*; do
    if [ -d "$f" ]; then
        for subdir in "$f"/*; do
            if [ -d "$subdir" ]; then
                directories+=("$subdir")
            fi
        done
    fi
done

for i in "${!directories[@]}"; do
    python3 animate.py -d ${directories[i]} -e 0 100 200 300 400 500 600 700 800 900 -s 8888 8988 9088 9188 9288 9388 9488 9588 9688 9788
done
