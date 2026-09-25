#!/usr/bin/env bash

set -e

# Accept either "vX.Y.Z" or "X.Y.Z"
VERSION="${1#v}"
TAG="v${VERSION}"

echo "Updating to version ${VERSION} (tag ${TAG})"

# Update pyproject.toml
sed -i.bak \
    -E "s/^version = \"[^\"]+\"/version = \"${VERSION}\"/" \
    pyproject.toml
rm pyproject.toml.bak

# Update source-file preambles
python3 h000_find_and_modify_preamble.py "${TAG}"

# Commit the version changes
git add pyproject.toml
git add PyECLOUD
git commit -m "Bump version to ${VERSION}"

# Create and push tag
git tag -a "${TAG}" -m "Bump version to ${VERSION}"
git push origin HEAD
git push origin "${TAG}"