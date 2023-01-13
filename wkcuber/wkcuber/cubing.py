import logging
from argparse import ArgumentParser
from os import path
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import numpy as np
from cluster_tools import Executor
from natsort import natsorted
from webknossos import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    BoundingBox,
    DataFormat,
    Dataset,
    Layer,
    Mag,
    SegmentationLayer,
    Vec3Int,
    View,
)
from webknossos.dataset._downsampling_utils import (
    InterpolationModes,
    downsample_unpadded_data,
    parse_interpolation_mode,
)
from webknossos.dataset.defaults import DEFAULT_CHUNK_SHAPE
from webknossos.utils import time_start, time_stop

from ._internal.image_readers import image_reader
from ._internal.utils import (
    add_batch_size_flag,
    add_data_format_flags,
    add_distribution_flags,
    add_interpolation_flag,
    add_verbose_flag,
    add_voxel_size_flag,
    find_files,
    get_chunks,
    get_executor_for_args,
    parse_path,
    setup_logging,
    setup_warnings,
    wait_and_ensure_success,
)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Directory containing the input images.", type=Path
    )

    parser.add_argument(
        "target_path",
        help="Output directory for the generated dataset.",
        type=parse_path,
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation)",
        default="color",
    )

    parser.add_argument(
        "--skip_first_z_slices",
        help="The number of z slices to skip at the beginning. This is useful to continue at a specific z, if a previous run was interrupted.",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--start_z",
        help="The offset of the first z slice.",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--dtype",
        "-d",
        help="Target datatype (e.g. uint8, uint16, uint32)",
        default=None,
    )

    add_batch_size_flag(parser)

    parser.add_argument(
        "--pad",
        help="Automatically pad image files at the bottom and right borders. "
        "Use this, when the input images don't have a common size, but have "
        "their origin at (0, 0).",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--target_mag",
        help="Automatically downsamples the cubed images to the provided "
        "magnification before writing to disk. The magnification can "
        "be provided like 2-2-1.",
        default="1",
        type=Mag,
    )

    parser.add_argument(
        "--channel_index",
        "-c",
        type=int,
        default=None,
        help="Select a single channel to be cubed into a layer. This can be helpful if several channels should be converted to multiple data layers. Note that WEBKNOSSOS only supports multiple channels if these are three uint8 channels. If this is not the case, use --channel_index to create a layer per channel.",
    )

    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Select a single sample of a specific channel to be cubed into a layer. This option is only valid if channel_index is set. Since WEBKNOSSOS only supports multiple uint8 channels, it may be necessary to cube a multi-sample dataset to different layers.",
    )

    add_voxel_size_flag(parser)
    add_interpolation_flag(parser)
    add_verbose_flag(parser)
    add_distribution_flags(parser)
    add_data_format_flags(parser)

    return parser


def find_source_filenames(source_path: Path) -> List[Path]:
    # Find all source files that have a matching file extension
    if source_path.is_dir():
        source_path_str = path.join(source_path, "*")
    else:
        source_path_str = str(source_path)

    source_files = list(find_files(source_path_str, image_reader.readers.keys()))

    assert len(source_files) > 0, (
        "No image files found in path "
        + source_path_str
        + ". Supported suffixes are "
        + str(image_reader.readers.keys())
        + "."
    )

    return natsorted(source_files)


def read_image_file(
    file_name: Path,
    dtype: np.dtype,
    z_slice: int,
    channel_index: Optional[int],
    sample_index: Optional[int],
) -> np.ndarray:
    try:
        return image_reader.read_array(
            file_name, dtype, z_slice, channel_index, sample_index
        )
    except Exception as exc:
        logging.error("Reading of file={} failed with {}".format(file_name, exc))
        raise exc


def cubing_job(
    args: Tuple[
        View,
        Mag,
        InterpolationModes,
        List[str],
        int,
        bool,
        Optional[int],
        Optional[int],
        str,
        int,
    ]
) -> Any:
    (
        target_view,
        target_mag,
        interpolation_mode,
        source_file_batches,
        batch_size,
        pad,
        channel_index,
        sample_index,
        dtype,
        num_channels,
    ) = args

    downsampling_needed = target_mag != Mag(1)
    largest_value_in_chunk = 0  # This is used to compute the largest_segmentation_id if it is a segmentation layer

    max_image_size = (target_view.size[0], target_view.size[1])

    # Iterate over batches of continuous z sections
    # The batches have a maximum size of `batch_size`
    # Batched iterations allows to utilize IO more efficiently
    first_z_idx = target_view.global_offset.z
    for source_file_batch in get_chunks(source_file_batches, batch_size):
        try:
            time_start(
                f"Cubing of z={first_z_idx}-{first_z_idx + len(source_file_batch)}"
            )
            # Allocate a large buffer for all images in this batch
            # Shape will be (channel_count, x, y, z)
            # Using fortran order for the buffer, prevents that the data has to be copied in rust
            buffer_shape = (
                [num_channels] + list(max_image_size) + [len(source_file_batch)]
            )
            buffer = np.empty(buffer_shape, dtype=dtype, order="F")

            # Iterate over each z section in the batch
            for i, file_name in enumerate(source_file_batch):
                z = first_z_idx + i
                # Image shape will be (x, y, channel_count, z=1)
                image = read_image_file(
                    file_name,
                    target_view.info.voxel_type,
                    z,
                    channel_index,
                    sample_index,
                )

                if pad:
                    image = np.pad(
                        image,
                        mode="constant",
                        pad_width=[
                            (0, max_image_size[0] - image.shape[0]),
                            (0, max_image_size[1] - image.shape[1]),
                            (0, 0),
                            (0, 0),
                        ],
                    )
                else:
                    assert (
                        image.shape[0:2] == max_image_size
                    ), "Section z={} has the wrong dimensions: {} (expected {}). Consider using --pad.".format(
                        z, image.shape, max_image_size
                    )
                buffer[:, :, :, i] = image.transpose((2, 0, 1, 3))[:, :, :, 0]
            del image

            if downsampling_needed:
                buffer = downsample_unpadded_data(
                    buffer, target_mag, interpolation_mode
                )
            buffer_z_offset = (
                first_z_idx - target_view.global_offset.z
            ) // target_mag.z
            target_view.write(offset=(0, 0, buffer_z_offset), data=buffer)
            largest_value_in_chunk = max(largest_value_in_chunk, np.max(buffer))
            time_stop(
                f"Cubing of z={first_z_idx}-{first_z_idx + len(source_file_batch)}"
            )
            first_z_idx += len(source_file_batch)

        except Exception as exc:
            logging.error(
                "Cubing of z={}-{} failed with {}".format(
                    first_z_idx, first_z_idx + len(source_file_batch), exc
                )
            )
            raise exc

    return largest_value_in_chunk


def get_channel_and_sample_count_and_dtype(source_path: Path) -> Tuple[int, int, str]:
    source_files = find_source_filenames(source_path)
    assert (
        len(source_files) > 0
    ), f"Failed to detect channel count and dtype. No sources found in {source_path}"
    return (
        image_reader.read_channel_count(source_files[0]),
        image_reader.read_sample_count(source_files[0]),
        image_reader.read_dtype(source_files[0]),
    )


def cubing(
    source_path: Path,
    target_path: Path,
    layer_name: str,
    batch_size: Optional[int],
    channel_index: Optional[int],
    sample_index: Optional[int],
    dtype: Optional[str],
    target_mag_str: str,
    data_format: DataFormat,
    chunk_shape: Vec3Int,
    chunks_per_shard: Vec3Int,
    interpolation_mode_str: str,
    start_z: int,
    skip_first_z_slices: int,
    pad: bool,
    voxel_size: Tuple[float, float, float],
    executor: Executor,
) -> Layer:
    source_files = find_source_filenames(source_path)

    all_num_x, all_num_y = zip(
        *[
            image_reader.read_dimensions(source_files[i])
            for i in range(len(source_files))
        ]
    )
    num_x = max(all_num_x)
    num_y = max(all_num_y)
    # All images are assumed to have equal channels and samples
    num_channels = image_reader.read_channel_count(source_files[0])
    num_samples = image_reader.read_sample_count(source_files[0])
    num_output_channels = num_channels * num_samples
    if channel_index is not None:
        # if there is no c axis, but someone meant to only use one channel/sample, set the sample index instead
        if sample_index is None and num_channels == 1 and channel_index > 0:
            sample_index = channel_index
            channel_index = 0

        assert (
            0 <= channel_index < num_channels
        ), "Selected channel is invalid. Please check the number of channels in the source file."
        num_output_channels = num_samples
    if sample_index is not None:
        # if no channel axis exists, it is valid to only set the sample index. Set channel index to 0 to avoid confusion
        if channel_index is None and num_channels == 1:
            channel_index = 0
        assert (
            channel_index is not None
        ), "Sample index is only valid if a channel index is also set."
        assert (
            0 <= sample_index < num_samples
        ), "Selected sample is invalid. Please check the number of samples in the source file."
        num_output_channels = 1
    num_z_slices_per_file = image_reader.read_z_slices_per_file(source_files[0])
    assert (
        num_z_slices_per_file == 1 or len(source_files) == 1
    ), "Multi page TIFF support only for single files"
    if num_z_slices_per_file > 1:
        num_z = num_z_slices_per_file
    else:
        num_z = len(source_files)

    if dtype is None:
        dtype = image_reader.read_dtype(source_files[0])

    if batch_size is None:
        batch_size = DEFAULT_CHUNK_SHAPE.z

    target_mag = Mag(target_mag_str)

    target_ds = Dataset(target_path, voxel_size=voxel_size, exist_ok=True)
    is_segmentation_layer = layer_name == "segmentation"

    if is_segmentation_layer:
        target_layer = target_ds.get_or_add_layer(
            layer_name,
            SEGMENTATION_CATEGORY,
            dtype_per_channel=dtype,
            num_channels=num_output_channels,
            largest_segment_id=0,
        )
    else:
        target_layer = target_ds.get_or_add_layer(
            layer_name,
            COLOR_CATEGORY,
            dtype_per_channel=dtype,
            num_channels=num_output_channels,
            data_format=data_format,
        )
    target_layer.bounding_box = target_layer.bounding_box.extended_by(
        BoundingBox(
            Vec3Int(0, 0, start_z + skip_first_z_slices) * target_mag,
            Vec3Int(num_x, num_y, num_z - skip_first_z_slices) * target_mag,
        )
    )

    target_mag_view = target_layer.get_or_add_mag(
        target_mag,
        chunks_per_shard=chunks_per_shard,
        chunk_shape=chunk_shape,
    )

    interpolation_mode = parse_interpolation_mode(
        interpolation_mode_str, target_layer.category
    )
    if target_mag != Mag(1):
        logging.info(
            f"Downsampling the cubed image to {target_mag} in memory with interpolation mode {interpolation_mode}."
        )

    logging.info("Found source files: count={} size={}x{}".format(num_z, num_x, num_y))

    job_args = []
    # We iterate over all z sections
    for z in range(skip_first_z_slices, num_z, DEFAULT_CHUNK_SHAPE.z):
        # The z is used to access the source files. However, when writing the data, the `start_z` has to be considered.
        max_z = min(num_z, z + DEFAULT_CHUNK_SHAPE.z)
        # Prepare source files array
        if len(source_files) > 1:
            source_files_array = source_files[z:max_z]
        else:
            source_files_array = source_files * (max_z - z)

        # Prepare job
        job_args.append(
            (
                target_mag_view.get_view(
                    (0, 0, z + start_z),
                    (num_x, num_y, max_z - z),
                ),
                target_mag,
                interpolation_mode,
                source_files_array,
                batch_size,
                pad,
                channel_index,
                sample_index,
                dtype,
                target_layer.num_channels,
            )
        )

        largest_segment_id_per_chunk = wait_and_ensure_success(
            executor.map_to_futures(cubing_job, job_args),
            progress_desc=f"Cubing from {skip_first_z_slices} to {num_z}",
        )
        if is_segmentation_layer:
            largest_segment_id = max(largest_segment_id_per_chunk)
            cast(
                SegmentationLayer, target_layer
            ).largest_segment_id = largest_segment_id

    # Return layer
    return target_layer


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    setup_warnings()

    arg_dict = vars(args)

    with get_executor_for_args(args) as executor:
        cubing(
            args.source_path,
            args.target_path,
            args.layer_name,
            arg_dict.get("batch_size"),
            arg_dict.get("channel_index"),
            arg_dict.get("sample_index"),
            arg_dict.get("dtype"),
            args.target_mag,
            args.data_format,
            args.chunk_shape,
            args.chunks_per_shard,
            args.interpolation_mode,
            args.start_z,
            args.skip_first_z_slices,
            args.pad,
            args.voxel_size,
            executor=executor,
        )
