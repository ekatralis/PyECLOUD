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
if ! mv Cross*.png cross_sections/comparison_plots/; then
    echo "Could not move Cross*.png into cross_sections/comparison_plots/" >> failed_tests.txt
fi

if [ -s failed_tests.txt ]; then
    echo "One or more tests failed; see failed_tests.txt"
    exit 1
fi

echo "All tests completed successfully."
