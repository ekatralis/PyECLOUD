#!/usr/bin/env bash

rm -f failed_tests.txt

for script in *.py; do
    for angle_dist in 2D 3D; do
        python "$script" --angle-dist-func "$angle_dist"
        if [ $? -ne 0 ]; then
            echo "python $script --angle-dist-func $angle_dist" >> failed_tests.txt
        fi
    done
done

mkdir -p cross_sections/comparison_plots
mv Cross*.png cross_sections/comparison_plots
