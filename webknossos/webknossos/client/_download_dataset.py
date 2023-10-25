import logging
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
from rich.progress import track

from ..dataset import Dataset, LayerCategoryType
from ..dataset.properties import LayerViewConfiguration, dataset_converter
from ..geometry import BoundingBox, Mag, Vec3Int
from ._generated.api.datastore import dataset_download
from ._generated.api.default import dataset_info
from ._generated.types import Unset
from .context import _get_context

logger = logging.getLogger(__name__)


T = TypeVar("T")


_DOWNLOAD_CHUNK_SHAPE = Vec3Int(512, 512, 512)


def download_dataset(
    dataset_name: str,
    organization_id: str,
    sharing_token: Optional[str] = None,
    bbox: Optional[BoundingBox] = None,
    layers: Optional[List[str]] = None,
    mags: Optional[List[Mag]] = None,
    path: Optional[Union[PathLike, str]] = None,
    exist_ok: bool = False,
) -> Dataset:
    context = _get_context()
    api_client = context.api_client
    api_dataset = api_client.dataset_info(
        organization_id, dataset_name, sharing_token=sharing_token
    )

    datastore_client = context.get_datastore_api_client(api_dataset.data_store.url)
    optional_datastore_token = sharing_token or context.datastore_token

    actual_path = Path(dataset_name) if path is None else Path(path)
    if actual_path.exists():
        logger.warning(f"{actual_path} already exists, skipping download.")
        return Dataset.open(actual_path)

    data_layers = api_dataset.data_source.data_layers
    voxel_size = api_dataset.data_source.scale
    if data_layers is None or len(data_layers) == 0 or voxel_size is None:
        raise RuntimeError(
            f"Could not download dataset {dataset_name}: {api_dataset.data_source.status or 'Unknown error.'}"
        )
    dataset = Dataset(
        actual_path, name=api_dataset.name, voxel_size=voxel_size, exist_ok=exist_ok
    )
    for layer_name in layers or [i.name for i in data_layers]:
        response_layers = [i for i in data_layers if i.name == layer_name]
        assert (
            len(response_layers) > 0
        ), f"The provided layer name {layer_name} could not be found in the requested dataset."
        assert (
            len(response_layers) == 1
        ), f"The provided layer name {layer_name} was found multiple times in the requested dataset."
        response_layer = response_layers[0]
        category = cast(LayerCategoryType, response_layer.category)
        layer = dataset.add_layer(
            layer_name=layer_name,
            category=category,
            dtype_per_layer=response_layer.element_class,
            num_channels=3 if response_layer.element_class == "uint24" else 1,
            largest_segment_id=response_layer.largest_segment_id,
        )

        # TODO view configuration
        default_view_configuration_dict = None
        if not isinstance(response_layer.default_view_configuration, Unset):
            default_view_configuration_dict = (
                response_layer.default_view_configuration.to_dict()
            )

        if default_view_configuration_dict is not None:
            default_view_configuration = dataset_converter.structure(
                default_view_configuration_dict, LayerViewConfiguration
            )
            layer.default_view_configuration = default_view_configuration

        if bbox is None:
            response_bbox = response_layer.bounding_box
            layer.bounding_box = BoundingBox(
                response_bbox.top_left,
                Vec3Int(response_bbox.width, response_bbox.height, response_bbox.depth),
            )
        else:
            assert isinstance(
                bbox, BoundingBox
            ), f"Expected a BoundingBox object for the bbox parameter but got {type(bbox)}"
            layer.bounding_box = bbox
        if mags is None:
            mags = [Mag(mag) for mag in response_layer.resolutions]
        for mag in mags:
            mag_view = layer.get_or_add_mag(
                mag,
                compress=True,
                chunk_shape=Vec3Int.full(32),
                chunks_per_shard=_DOWNLOAD_CHUNK_SHAPE // 32,
            )
            aligned_bbox = layer.bounding_box.align_with_mag(mag, ceil=True)
            download_chunk_shape_in_mag = _DOWNLOAD_CHUNK_SHAPE * mag.to_vec3_int()
            for chunk in track(
                list(
                    aligned_bbox.chunk(
                        download_chunk_shape_in_mag, download_chunk_shape_in_mag
                    )
                ),
                description=f"Downloading layer={layer.name} mag={mag}",
            ):
                chunk_in_mag = chunk.in_mag(mag)
                # TODO actual download
                response = dataset_download.sync_detailed(
                    organization_name=organization_id,
                    data_set_name=dataset_name,
                    data_layer_name=layer_name,
                    mag=mag.to_long_layer_name(),
                    client=datastore_client,
                    token=optional_datastore_token,
                    x=chunk.topleft.x,
                    y=chunk.topleft.y,
                    z=chunk.topleft.z,
                    width=chunk_in_mag.size.x,
                    height=chunk_in_mag.size.y,
                    depth=chunk_in_mag.size.z,
                )
                assert response.status_code == 200, response
                assert (
                    response.headers["missing-buckets"] == "[]"
                ), f"Download contained missing buckets {response.headers['missing-buckets']}."
                data = np.frombuffer(
                    response.content, dtype=layer.dtype_per_channel
                ).reshape(layer.num_channels, *chunk_in_mag.size, order="F")
                mag_view.write(data, absolute_offset=chunk.topleft)
    return dataset
