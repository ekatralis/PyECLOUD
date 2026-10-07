#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

STASHED=false
DIST_DIR=""

cleanup() {
    exit_code=$?

    # Disable errexit during cleanup so one cleanup failure doesn't
    # prevent the remaining cleanup steps.
    set +e

    if [[ -n "$DIST_DIR" && -d "$DIST_DIR" ]]; then
        rm -rf -- "$DIST_DIR"
    fi

    if [[ "$STASHED" == true ]]; then
        echo "Restoring stashed working tree..."
        git stash pop
        if [[ $? -ne 0 ]]; then
            echo "WARNING: Could not cleanly restore stashed changes."
            echo "The stash has been kept; resolve conflicts manually."
        fi
    fi

    exit "$exit_code"
}

trap cleanup EXIT

NAME=$(python setup.py --name)
VER=$(python setup.py --version)

echo "========================================================================"
echo "Bumping $NAME scripts to v$VER"
echo "========================================================================"

if [[ -n "$(git status --porcelain)" ]]; then
    git stash push --include-untracked -m "pre-release-$VER"
    STASHED=true
fi

python pre-release_update_preamble.py "v$VER"
git add PyECLOUD
git commit -m "Release v$VER"
git push

echo "========================================================================"
echo "Tagging $NAME v$VER"
echo "========================================================================"

git tag "v$VER"
git push origin "v$VER"

echo "========================================================================"
echo "Releasing $NAME v$VER on PyPI"
echo "========================================================================"

DIST_DIR=$(mktemp -d)
python setup.py sdist --dist-dir "$DIST_DIR"
twine upload "$DIST_DIR"/*.tar.gz