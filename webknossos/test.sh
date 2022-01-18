#!/usr/bin/env bash
set -eEuo pipefail

if [ $# -eq 1 ] && [ "$1" = "--refresh-snapshots" ]; then
    if ! curl -sf localhost:9000/api/health; then
        WK_DOCKER_DIR="tests"
        pushd $WK_DOCKER_DIR > /dev/null
        export DOCKER_TAG=master__16396
        docker-compose pull webknossos
        # TODO: either remove pg/db before starting or run tools/postgres/apply_evolutions.sh
        USER_UID=$(id -u) USER_GID=$(id -g) docker-compose up -d --no-build webknossos
        popd > /dev/null
        stop_wk () {
            ARG=$?
            pushd $WK_DOCKER_DIR > /dev/null
            docker-compose down
            popd > /dev/null
            exit $ARG
        } 
        trap stop_wk EXIT
        while ! curl -sf localhost:9000/api/health; do
            sleep 5
        done
    fi
    rm -rf tests/cassettes

    # Note that pytest should be executed via `python -m`, since
    # this will ensure that the current directory is added to sys.path
    # (which is standard python behavior). This is necessary so that the imports
    # refer to the checked out (and potentially modified) code.
    poetry run python -m pytest --record-mode once
else
    poetry run python -m pytest --block-network
fi
