import warnings
from pathlib import Path
from shutil import copy
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import gmtime, strftime
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from zipfile import BadZipFile, ZipFile

import httpx
import numpy as np
import pytest
from tests.constants import TESTDATA_DIR
from tifffile import TiffFile

import webknossos as wk

pytestmark = [pytest.mark.block_network(allowed_hosts=[".*"])]


@pytest.fixture(autouse=True, scope="function")
def ignore_warnings() -> Iterator:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="webknossos", message=r"\[WARNING\]")
        yield


def test_compare_tifffile(tmp_path: Path) -> None:
    ds = wk.Dataset(tmp_path, (1, 1, 1))
    l = ds.add_layer_from_images(
        "testdata/tiff/test.02*.tiff",
        layer_name="compare_tifffile",
        compress=True,
        category="segmentation",
        topleft=(100, 100, 55),
        chunk_shape=(8, 8, 8),
        chunks_per_shard=(8, 8, 8),
    )
    assert l.bounding_box.topleft == wk.Vec3Int(100, 100, 55)
    data = l.get_finest_mag().read()[0, :, :]
    for z_index in range(0, data.shape[-1]):
        with TiffFile("testdata/tiff/test.0200.tiff") as tif_file:
            comparison_slice = tif_file.asarray().T
        assert np.array_equal(data[:, :, z_index], comparison_slice)


def test_compare_nd_tifffile(tmp_path: Path) -> None:
    ds = wk.Dataset(tmp_path, (1, 1, 1))
    l = ds.add_layer_from_images(
        "testdata/4D/4D_series/4D-series.ome.tif",
        layer_name="color",
        category="color",
        topleft=(100, 100, 55),
        use_bioformats=True,
        data_format="zarr3",
        chunk_shape=(8, 8, 8),
        chunks_per_shard=(8, 8, 8),
    )
    assert l.bounding_box.topleft == wk.VecInt(0, 55, 100, 100)
    assert l.bounding_box.size == wk.VecInt(7, 5, 167, 439)
    read_with_tifffile_reader = TiffFile(
        "testdata/4D/4D_series/4D-series.ome.tif"
    ).asarray()
    read_first_channel_from_dataset = l.get_finest_mag().read()[0]
    assert np.array_equal(read_with_tifffile_reader, read_first_channel_from_dataset)


REPO_IMAGES_ARGS: List[
    Tuple[Union[str, List[Path]], Dict[str, Any], str, int, Tuple[int, int, int]]
] = [
    (
        "testdata/tiff/test.*.tiff",
        {"category": "segmentation"},
        "uint8",
        1,
        (265, 265, 257),
    ),
    (
        [
            TESTDATA_DIR / "tiff" / "test.0000.tiff",
            TESTDATA_DIR / "tiff" / "test.0001.tiff",
            TESTDATA_DIR / "tiff" / "test.0002.tiff",
        ],
        {},
        "uint8",
        1,
        (265, 265, 3),
    ),
    (
        "testdata/rgb_tiff/test_rgb.tif",
        {"mag": 2},
        "uint8",
        3,
        (64, 64, 2),
    ),
    (
        "testdata/rgb_tiff",
        {"mag": 2, "channel": 1, "dtype": "uint32"},
        "uint32",
        1,
        (64, 64, 2),
    ),
    (
        "testdata/temca2/*/*/*.jpg",
        {"flip_x": True, "batch_size": 2048},
        "uint8",
        1,
        (1024, 1024, 12),
    ),
    (
        "testdata/temca2",
        {"flip_z": True, "batch_size": 2048},
        "uint8",
        1,
        # The topmost folder contains an extra image,
        # which is included here as well, but not in
        # the glob pattern above. Therefore z is +1.
        (1024, 1024, 13),
    ),
    (
        "testdata/tiff_with_different_shapes/*",
        {"flip_y": True},
        "uint8",
        1,
        (2970, 2521, 4),
    ),
    ("testdata/various_tiff_formats/test_CS.tif", {}, "uint8", 3, (128, 128, 320)),
    ("testdata/various_tiff_formats/test_C.tif", {}, "uint8", 1, (128, 128, 320)),
    # same as test_C.tif above, but as a single file in a folder:
    ("testdata/single_multipage_tiff_folder", {}, "uint8", 1, (128, 128, 320)),
    ("testdata/various_tiff_formats/test_I.tif", {}, "uint32", 1, (64, 128, 64)),
    ("testdata/various_tiff_formats/test_S.tif", {}, "uint16", 3, (128, 128, 64)),
]


@pytest.mark.parametrize("path, kwargs, dtype, num_channels, size", REPO_IMAGES_ARGS)
def test_repo_images(
    tmp_path: Path,
    path: str,
    kwargs: Dict,
    dtype: str,
    num_channels: int,
    size: Tuple[int, int, int],
) -> wk.Dataset:
    with wk.utils.get_executor_for_args(None) as executor:
        ds = wk.Dataset(tmp_path, (1, 1, 1))
        layer_name = "__".join(
            (path if isinstance(path, str) else str(path[0])).split("/")[1:]
        )
        l = ds.add_layer_from_images(
            path,
            layer_name=layer_name,
            compress=True,
            executor=executor,
            use_bioformats=False,
            **kwargs,
        )
        assert l.dtype_per_channel == np.dtype(dtype)
        assert l.num_channels == num_channels
        assert l.bounding_box == wk.BoundingBox(topleft=(0, 0, 0), size=size)
        if isinstance(l, wk.SegmentationLayer):
            assert l.largest_segment_id is not None
            assert l.largest_segment_id > 0
    return ds


def download_and_unpack(
    url: Union[str, List[str]], out_path: Path, filename: Union[str, List[str]]
) -> None:
    if isinstance(url, str):
        assert isinstance(filename, str)
        url = [url]
        filename = [filename]
    for url_i, filename_i in zip(url, filename):
        with NamedTemporaryFile() as download_file:
            with httpx.stream("GET", url_i, follow_redirects=True) as response:
                total = int(response.headers["Content-Length"])

                with wk.utils.get_rich_progress() as progress:
                    download_task = progress.add_task(
                        "Download Image Data", total=total
                    )
                    for chunk in response.iter_bytes():
                        download_file.write(chunk)
                        progress.update(
                            download_task, completed=response.num_bytes_downloaded
                        )
            try:
                with ZipFile(download_file, "r") as zip_file:
                    zip_file.extractall(out_path)
            except BadZipFile:
                out_path.mkdir(parents=True, exist_ok=True)
                copy(download_file.name, out_path / filename_i)


BIOFORMATS_ARGS = [
    (
        "https://samples.scif.io/wtembryo.zip",
        "wtembryo.mov",
        {},
        "uint8",
        3,
        (320, 240, 108),
        1,
    ),
    (
        "https://samples.scif.io/wtembryo.zip",
        "wtembryo.mov",
        {"timepoint": 50, "swap_xy": True},
        "uint8",
        3,
        (240, 320, 1),
        1,
    ),
    (
        "https://samples.scif.io/HEART.zip",
        "HEART.SEQ",
        {"flip_z": True},
        "uint8",
        1,
        (512, 512, 30),
        1,
    ),
    (
        "https://samples.scif.io/sdub.zip",
        "sdub*.pic",
        {"timepoint": 3},
        "uint8",
        1,
        (192, 128, 9),
        1,
    ),
    (
        "https://samples.scif.io/test-avi.zip",
        "t1-rendering.avi",
        {},
        "uint8",
        3,
        (206, 218, 36),
        1,
    ),
]


@pytest.mark.parametrize(
    "url, filename, kwargs, dtype, num_channels, size, num_layers", BIOFORMATS_ARGS
)
def test_bioformats(
    tmp_path: Path,
    url: str,
    filename: str,
    kwargs: Dict,
    dtype: str,
    num_channels: int,
    size: Tuple[int, int, int],
    num_layers: int,
) -> wk.Dataset:
    unzip_path = tmp_path / "unzip"
    download_and_unpack(url, unzip_path, filename)
    ds = wk.Dataset(tmp_path / "ds", (1, 1, 1))
    with wk.utils.get_executor_for_args(None) as executor:
        l = ds.add_layer_from_images(
            str(unzip_path / filename),
            layer_name=filename,
            compress=True,
            executor=executor,
            use_bioformats=True,
            **kwargs,
        )
        assert l.dtype_per_channel == np.dtype(dtype)
        assert l.num_channels == num_channels
        assert l.bounding_box == wk.BoundingBox(topleft=(0, 0, 0), size=size)
    assert len(ds.layers) == num_layers
    return ds


# All scif images used here are published with CC0 license,
# see https://scif.io/images.
TEST_IMAGES_ARGS = [
    (
        "https://static.webknossos.org/data/webknossos-libs/slice_0420.dm4",
        "slice_0420.dm4",
        {"data_format": "zarr"},  # using zarr to allow z=1 chunking
        "uint16",
        1,
        (8192, 8192, 1),
    ),
    (
        "https://static.webknossos.org/data/webknossos-libs/slice_0073.dm3",
        "slice_0073.dm3",
        {"data_format": "zarr"},  # using zarr to allow z=1 chunking
        "uint16",
        1,
        (4096, 4096, 1),
    ),
    (
        [
            "https://static.webknossos.org/data/webknossos-libs/slice_0073.dm3",
            "https://static.webknossos.org/data/webknossos-libs/slice_0074.dm3",
        ],
        ["slice_0073.dm3", "slice_0074.dm3"],
        {"data_format": "zarr"},  # using zarr to allow smaller chunking
        "uint16",
        1,
        (4096, 4096, 2),
    ),
    (
        "https://samples.scif.io/dnasample1.zip",
        "dnasample1.dm3",
        {"data_format": "zarr"},  # using zarr to allow z=1 chunking
        "int16",
        1,
        (4096, 4096, 1),
    ),
    (
        # published with CC0 license, taken from
        # https://doi.org/10.6084/m9.figshare.c.3727411_D391.v1
        "https://figshare.com/ndownloader/files/8909407",
        "embedded_NCI_mono_matrigelcollagen_docetaxel_day10_sample10.czi",
        {},
        "uint16",
        1,
        (512, 512, 30),
    ),
    (
        "https://samples.scif.io/test-gif.zip",
        "scifio-test.gif",
        {},
        "uint8",
        3,
        (500, 500, 1),
    ),
    (
        "https://samples.scif.io/test-jpeg2000.zip",
        "scifio-test.jp2",
        {},
        "uint8",
        3,
        (500, 500, 1),
    ),
    (
        "https://samples.scif.io/test-jpg.zip",
        "scifio-test.jpg",
        {"flip_x": True, "batch_size": 2048},
        "uint8",
        3,
        (500, 500, 1),
    ),
    (
        "https://samples.scif.io/test-png.zip",
        "scifio-test.png",
        {"flip_y": True},
        "uint8",
        3,
        (500, 500, 1),
    ),
]


@pytest.mark.parametrize(
    "url, filename, kwargs, dtype, num_channels, size", TEST_IMAGES_ARGS
)
def test_test_images(
    tmp_path: Path,
    url: Union[str, List[str]],
    filename: Union[str, List[str]],
    kwargs: Dict,
    dtype: str,
    num_channels: int,
    size: Tuple[int, int, int],
) -> wk.Dataset:
    unzip_path = tmp_path / "unzip"
    download_and_unpack(url, unzip_path, filename)
    path: Union[Path, List[Path]]
    if isinstance(filename, list):
        layer_name = filename[0] + "..."
        path = [unzip_path / i for i in filename]
    else:
        layer_name = filename
        path = unzip_path / filename
    ds = wk.Dataset(tmp_path / "ds", (1, 1, 1))
    with wk.utils.get_executor_for_args(None) as executor:
        l_bio: Optional[wk.Layer]
        try:
            l_bio = ds.add_layer_from_images(
                path,
                layer_name="bioformats_" + layer_name,
                compress=True,
                executor=executor,
                use_bioformats=True,
                **kwargs,
            )
        except Exception as e:
            print(e)
            l_bio = None
        else:
            assert l_bio.dtype_per_channel == np.dtype(dtype)
            assert l_bio.num_channels == num_channels
            assert l_bio.bounding_box == wk.BoundingBox(topleft=(0, 0, 0), size=size)
        l_normal = ds.add_layer_from_images(
            path,
            layer_name="normal_" + layer_name,
            compress=True,
            executor=executor,
            use_bioformats=False,
            **kwargs,
        )
        assert l_normal.dtype_per_channel == np.dtype(dtype)
        assert l_normal.num_channels == num_channels
        assert l_normal.bounding_box == wk.BoundingBox(topleft=(0, 0, 0), size=size)
        if l_bio is not None:
            assert np.array_equal(
                l_bio.get_finest_mag().read(), l_normal.get_finest_mag().read()
            )
    return ds


if __name__ == "__main__":
    time = lambda: strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    for repo_images_args in REPO_IMAGES_ARGS:
        with TemporaryDirectory() as tempdir:
            image_path = repo_images_args[0]
            if isinstance(image_path, list):
                image_path = str(image_path[0])
            name = "".join(filter(str.isalnum, image_path))
            print(*repo_images_args)
            print(
                test_repo_images(Path(tempdir), *repo_images_args)
                .upload(f"test_repo_images_{name}_{time()}")
                .url
            )

    for bioformats_args in BIOFORMATS_ARGS:
        with TemporaryDirectory() as tempdir:
            name = "".join(filter(str.isalnum, bioformats_args[1]))
            print(*bioformats_args)
            print(
                test_bioformats(Path(tempdir), *bioformats_args)
                .upload(f"test_bioformats_{name}_{time()}")
                .url
            )

    for test_images_args in TEST_IMAGES_ARGS:
        with TemporaryDirectory() as tempdir:
            name = "".join(filter(str.isalnum, test_images_args[1]))
            print(*test_images_args)
            print(
                test_test_images(Path(tempdir), *test_images_args)
                .upload(f"test_test_images_{name}_{time()}")
                .url
            )
