from pathlib import Path
from tempfile import TemporaryDirectory
from time import gmtime, strftime
from typing import Iterator

import numpy as np
import pytest

import webknossos as wk

SAMPLE_BBOX = wk.BoundingBox((3164, 3212, 1017), (10, 10, 10))

pytestmark = [pytest.mark.use_proxay]


@pytest.fixture(scope="module")
def sample_dataset() -> Iterator[wk.Dataset]:
    yield wk.Dataset.open(
        Path(__file__).parent.parent.parent / "testdata" / "l4_sample_snipped"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:9000/datasets/Organization_X/l4_sample",
        "http://localhost:9000/datasets/Organization_X/l4_sample/view",
        # "http://localhost:9000/datasets/scalable_minds/l4_sample_dev_sharing/view?token=ilDXmfQa2G8e719vb1U9YQ#%7B%22orthogonal%7D",
        # "http://localhost:9000/links/93zLg9U9vJ3c_UWp",
    ],
)
def test_url_download(url: str, tmp_path: Path, sample_dataset: wk.Dataset) -> None:
    ds = wk.Dataset.download(
        url, path=tmp_path / "ds", mags=[wk.Mag(1)], bbox=SAMPLE_BBOX
    )
    assert set(ds.layers.keys()) == {"color", "segmentation"}
    data = ds.get_color_layers()[0].get_finest_mag().read()
    assert data.sum() == 120697
    assert np.array_equal(
        data,
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:9000/datasets/Organization_X/l4_sample",
        "http://localhost:9000/datasets/Organization_X/l4_sample/view",
        # "http://localhost:9000/datasets/Organization_X/l4_sample_dev_sharing/view?token=ilDXmfQa2G8e719vb1U9YQ#%7B%22orthogonal%7D",
        # "http://localhost:9000/links/93zLg9U9vJ3c_UWp",
    ],
)
def test_url_open_remote(url: str, sample_dataset: wk.Dataset) -> None:
    ds = wk.Dataset.open_remote(
        url,
    )
    assert set(ds.layers.keys()) == {"color", "segmentation"}
    data = (
        ds.get_color_layers()[0]
        .get_finest_mag()
        .read(absolute_bounding_box=SAMPLE_BBOX)
    )
    assert data.sum() == 120697
    assert np.array_equal(
        data,
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )


@pytest.mark.use_proxay
def test_remote_dataset(sample_dataset: wk.Dataset) -> None:
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    remote_ds = sample_dataset.upload(
        new_dataset_name=f"test_remote_metadata_{time_str}"
    )
    assert np.array_equal(
        remote_ds.get_color_layers()[0].get_finest_mag().read(),
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )

    assert remote_ds.read_only
    assert remote_ds.get_color_layers()[0].read_only
    assert remote_ds.get_color_layers()[0].get_finest_mag().read_only

    assert (
        remote_ds.url
        == f"http://localhost:9000/datasets/Organization_X/test_remote_metadata_{time_str}"
    )

    assert remote_ds.display_name is None
    remote_ds.display_name = "Test Remote Dataset"
    assert remote_ds.display_name == "Test Remote Dataset"
    del remote_ds.display_name
    assert remote_ds.display_name is None

    assert remote_ds.description is None
    remote_ds.description = "My awesome test description"
    assert remote_ds.description == "My awesome test description"
    del remote_ds.description
    assert remote_ds.description is None

    assert not remote_ds.is_public
    remote_ds.is_public = True
    assert remote_ds.is_public

    assert len(remote_ds.tags) == 0
    for i in range(3):
        remote_ds.tags += (f"category {i}",)
    assert remote_ds.tags == ("category 0", "category 1", "category 2")

    assert len(remote_ds.sharing_token) > 0

    assert len(remote_ds.allowed_teams) == 0
    test_teams = (wk.Team.get_by_name("team_X1"),)
    assert test_teams[0].id == "570b9f4b2a7c0e3b008da6ec"
    remote_ds.allowed_teams = test_teams
    assert remote_ds.allowed_teams == test_teams
    remote_ds.allowed_teams = ["570b9f4b2a7c0e3b008da6ec"]
    assert remote_ds.allowed_teams == test_teams
    remote_ds.folder = wk.RemoteFolder.get_by_path("Organization_X/A subfolder!")
    assert remote_ds.folder.name == "A subfolder!"


@pytest.mark.use_proxay
def test_upload_download_roundtrip(sample_dataset: wk.Dataset, tmp_path: Path) -> None:
    ds_original = sample_dataset
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    url = ds_original.upload(
        new_dataset_name=f"test_upload_download_roundtrip_{time_str}"
    ).url
    ds_roundtrip = wk.Dataset.download(
        url, path=tmp_path / "ds", layers=["color", "segmentation"]
    )
    assert set(ds_original.get_segmentation_layers()[0].mags.keys()) == set(
        ds_roundtrip.get_segmentation_layers()[0].mags.keys()
    )

    original_config = ds_original.get_layer("color").default_view_configuration
    roundtrip_config = ds_roundtrip.get_layer("color").default_view_configuration
    assert (
        original_config is not None
    ), "default_view_configuration should be defined for original dataset"
    assert (
        roundtrip_config is not None
    ), "default_view_configuration should be defined for roundtrip dataset"
    assert original_config.color == roundtrip_config.color
    assert original_config.intensity_range == roundtrip_config.intensity_range

    data_original = ds_original.get_segmentation_layers()[0].get_finest_mag().read()
    data_roundtrip = ds_roundtrip.get_segmentation_layers()[0].get_finest_mag().read()
    assert np.array_equal(data_original, data_roundtrip)
