#!/usr/bin/env bash
set -eEuo pipefail
set +x

for PKG in */pyproject.toml; do
    PKG="$(dirname "$PKG")"
    pushd "$PKG" > /dev/null

    cp pyproject.toml pyproject.toml.bak
    PKG_VERSION="$(dunamai from git)"

    poetry version "$PKG_VERSION"
    # replace all relative path dependencies with the current version:
    sed -i 's/\(.*\) = .* path \= \"\.\..*/\1 = "'"$PKG_VERSION"'"/g' pyproject.toml
    poetry publish --build -u "$PYPI_USERNAME" -p "$PYPI_PASSWORD"

    # Restore files
    mv pyproject.toml.bak pyproject.toml

    popd > /dev/null
