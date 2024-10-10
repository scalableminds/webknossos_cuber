#! /usr/bin/env bash
set -Eeo pipefail

if [ ! -d "wk-repo" ]; then
    echo
    echo ERROR!
    echo 'Either link or clone the webknossos repository to "docs/wk-repo", e.g. with'
    echo 'git clone --depth 1 git@github.com:scalableminds/webknossos.git docs/wk-repo'
    exit 1
fi
rm -rf src/api/webknossos
PYTHONPATH=$PYTHONPATH uv run python generate_api_doc_pages.py

if [ $# -eq 1 ] && [ "$1" = "--persist" ]; then
    PYTHONPATH=$PYTHONPATH uv run mkdocs build
else
    PYTHONPATH=$PYTHONPATH uv run mkdocs serve -a localhost:8197 --watch-theme
fi
