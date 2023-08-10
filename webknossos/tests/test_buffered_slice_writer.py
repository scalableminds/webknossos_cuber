import difflib
from os import PathLike
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

import webknossos as wk
from webknossos.utils import time_start, time_stop

from .constants import TESTDATA_DIR


def test_basic_buffered_slice_writer() -> None:
    # Create DS
    dataset = wk.Dataset("testoutput/bsw_test", voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color", category="color", dtype_per_channel="uint8", num_channels=1
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), chunks_per_shard=(8, 8, 8))

    # Allocate some data (~ 8 MB)
    shape = (512, 512, 32)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    # Write some slices
    time_start("write")
    with mag1.get_buffered_slice_writer() as writer:
        for z in range(0, shape[2]):
            section = data[:, :, z]
            writer.send(section)
    time_stop("write")
    print("write done")

    time_start("read")
    written_data = mag1.read(absolute_offset=(0, 0, 0), size=shape)
    time_stop("read")

    time_start("compare")
    assert np.all(data == written_data)
    time_stop("compare")


def test_basic_buffered_slice_writer_multi_shard() -> None:
    # Create DS
    dataset = wk.Dataset("testoutput/bsw_test", voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color", category="color", dtype_per_channel="uint8", num_channels=1
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), chunks_per_shard=(4, 4, 4))

    # Allocate some data (~ 3 MB) that covers multiple shards (also in z)
    shape = (160, 150, 140)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    # Write some slices
    time_start("write")
    with mag1.get_buffered_slice_writer() as writer:
        for z in range(0, shape[2]):
            section = data[:, :, z]
            writer.send(section)
    time_stop("write")
    print("write done")

    time_start("read")
    written_data = mag1.read(absolute_offset=(0, 0, 0), size=shape)
    time_stop("read")

    time_start("compare")
    assert np.all(data == written_data)
    time_stop("compare")


def test_basic_buffered_slice_writer_multi_shard_multi_channel() -> None:
    # Create DS
    dataset = wk.Dataset("testoutput/bsw_test", voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color", category="color", dtype_per_channel="uint8", num_channels=3
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), chunks_per_shard=(4, 4, 4))

    # Allocate some data (~ 3 MB) that covers multiple shards (also in z)
    shape = (3, 160, 150, 140)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    # Write some slices
    time_start("write")
    with mag1.get_buffered_slice_writer() as writer:
        for z in range(0, shape[-1]):
            section = data[:, :, :, z]
            writer.send(section)
    time_stop("write")
    print("write done")

    time_start("read")
    written_data = mag1.read(absolute_offset=(0, 0, 0), size=shape[1:])
    time_stop("read")

    time_start("compare")
    assert np.all(data == written_data)
    time_stop("compare")
