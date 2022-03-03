import logging
import warnings
from pathlib import Path
from typing import Tuple

from webknossos.dataset.storage import WKWStorageArray

from ..utils import time_start, time_stop


def compress_file_job(args: Tuple[Path, Path]) -> None:
    warnings.warn("[DEPRECATION] compress_file_job is deprecated.")
    source_path, target_path = args
    try:
        time_start("Compressing '{}' to '{}'".format(source_path, target_path))

        target_path.parent.mkdir(parents=True, exist_ok=True)
        WKWStorageArray.compress_shard(source_path, target_path)

        if not target_path.exists():
            raise Exception("Did not create compressed file {}".format(target_path))

        time_stop("Compressing '{}' to '{}'".format(source_path, target_path))

    except Exception as exc:
        logging.error(
            "Compression of '{}' to '{}' failed with {}".format(
                source_path, target_path, exc
            )
        )
        raise exc
