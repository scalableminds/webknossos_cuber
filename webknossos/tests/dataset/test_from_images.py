import warnings
from pathlib import Path
from shutil import copytree
from typing import Iterator

import numpy as np
import pytest
from tests.constants import TESTDATA_DIR
from tifffile import TiffFile

import webknossos as wk
from webknossos.dataset import Dataset


@pytest.fixture(autouse=True, scope="function")
def ignore_warnings() -> Iterator:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="webknossos", message=r"\[WARNING\]")
        yield


def test_compare_tifffile(tmp_path: Path) -> None:
    ds = wk.Dataset.from_images(
        TESTDATA_DIR / "tiff",
        tmp_path,
        (1, 1, 1),
        compress=True,
        layer_category="segmentation",
        chunks_per_shard=(8, 8, 8),
        map_filepath_to_layer_name=wk.Dataset.ConversionLayerMapping.ENFORCE_SINGLE_LAYER,
    )
    assert len(ds.layers) == 1
    assert "tiff" in ds.layers
    data = ds.layers["tiff"].get_finest_mag().read()[0, :, :]
    for z_index in range(0, data.shape[-1]):
        with TiffFile(TESTDATA_DIR / "tiff" / "test.0000.tiff") as tif_file:
            comparison_slice = tif_file.asarray().T
        assert np.array_equal(data[:, :, z_index], comparison_slice)


def test_multiple_multitiffs(tmp_path: Path) -> None:
    ds = wk.Dataset.from_images(
        TESTDATA_DIR / "various_tiff_formats",
        tmp_path,
        (1, 1, 1),
    )
    assert len(ds.layers) == 4

    expected_dtype_channels_size_per_layer = {
        "test_CS.tif": ("uint8", 3, (128, 128, 320)),
        "test_C.tif": ("uint8", 1, (128, 128, 320)),
        "test_I.tif": ("uint32", 1, (64, 128, 64)),
        "test_S.tif": ("uint16", 3, (128, 128, 64)),
    }

    for layer_name, layer in ds.layers.items():
        dtype, channels, size = expected_dtype_channels_size_per_layer[layer_name]
        assert layer.dtype_per_channel == np.dtype(dtype)
        assert layer.num_channels == channels
        assert layer.bounding_box.size == size


def test_from_dicom_images(tmp_path: Path) -> None:
    ds = wk.Dataset.from_images(
        TESTDATA_DIR / "dicoms",
        tmp_path,
        (1, 1, 1),
    )
    assert len(ds.layers) == 1
    assert "dicoms" in ds.layers
    data = ds.layers["dicoms"].get_finest_mag().read()
    assert data.shape == (1, 274, 384, 384)
    assert (
        data.max() == 127
    ), f"The maximum value of the image should be 127 but is {data.max()}"


def test_no_slashes_in_layername(tmp_path: Path) -> None:
    (input_path := tmp_path / "tiff" / "subfolder" / "tifffiles").mkdir(parents=True)
    copytree(
        TESTDATA_DIR / "tiff_with_different_shapes",
        input_path,
        dirs_exist_ok=True,
    )

    for strategy in Dataset.ConversionLayerMapping:
        dataset = wk.Dataset.from_images(
            tmp_path / "tiff",
            tmp_path / str(strategy),
            voxel_size=(10, 10, 10),
            map_filepath_to_layer_name=strategy,
        )

        assert all("/" not in layername for layername in dataset.layers)
