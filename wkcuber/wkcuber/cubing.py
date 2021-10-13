import time
import logging
from typing import List, Tuple, Optional, cast, Any

import numpy as np
from argparse import ArgumentParser, Namespace
from os import path
from pathlib import Path
from natsort import natsorted

from webknossos.dataset import Dataset, LayerCategories, View, SegmentationLayer, Layer
from webknossos.geometry import BoundingBox, Vec3Int
from .mag import Mag
from .downsampling_utils import (
    parse_interpolation_mode,
    downsample_unpadded_data,
    InterpolationModes,
)
from .utils import (
    get_chunks,
    find_files,
    add_batch_size_flag,
    add_verbose_flag,
    add_distribution_flags,
    add_interpolation_flag,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
)
from .image_readers import image_reader, refresh_global_image_reader

BLOCK_LEN = 32


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Directory containing the input images.", type=Path
    )

    parser.add_argument(
        "target_path", help="Output directory for the generated dataset.", type=Path
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation)",
        default="color",
    )

    parser.add_argument(
        "--start_z",
        help="The z coordinate of the first slice. This is useful to continue at a specific z, if a previous run was interrupted.",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--dtype",
        "-d",
        help="Target datatype (e.g. uint8, uint16, uint32)",
        default=None,
    )

    parser.add_argument(
        "--wkw_file_len",
        default=32,
        type=int,
        help="Amount of blocks which are written per dimension to a wkw cube. The default value of 32 means that 1024 slices are written to one cube (since one block has 32**3 voxels by default). For single-channel uint8 data, this results in 1 GB per cube file. If file_len is set to 1, only 32 slices are written to one cube. Must be a power of two.",
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
        help="Select a single channel to be cubed into a layer. This can be helpful if several channels should be converted to multiple data layers. Note that webKnossos only supports multiple channels if these are three uint8 channels. If this is not the case, use --channel_index to create a layer per channel.",
    )

    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Select a single sample of a specific channel to be cubed into a layer. This option is only valid if channel_index is set. Since webKnossos only supports multiple uint8 channels, it may be necessary to cube a multi-sample dataset to different layers.",
    )

    add_interpolation_flag(parser)
    add_verbose_flag(parser)
    add_distribution_flags(parser)

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
    dtype: type,
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


def prepare_slices_for_wkw(
    slices: List[np.ndarray], num_channels: int = None
) -> np.ndarray:
    # Write batch buffer which will have shape (x, y, channel_count, z)
    # since we concat along the last axis (z)
    buffer = np.concatenate(slices, axis=-1)

    # We transpose the data so that the first dimension is the channel,
    # since the wkw library expects this.
    # New shape will be (channel_count, x, y, z)
    buffer = np.transpose(buffer, (2, 0, 1, 3))
    if num_channels is not None:
        assert buffer.shape[0] == num_channels
    return buffer


def cubing_job(
    args: Tuple[
        View,
        Mag,
        InterpolationModes,
        List[str],
        int,
        Tuple[int, int],
        bool,
        Optional[int],
        Optional[int],
    ]
) -> Any:
    (
        target_view,
        target_mag,
        interpolation_mode,
        source_file_batches,
        batch_size,
        image_size,
        pad,
        channel_index,
        sample_index,
    ) = args

    downsampling_needed = target_mag != Mag(1)
    largest_value_in_chunk = 0  # This is used to compute the largest_segmentation_id if it is a segmentation layer

    with target_view.open():
        # Iterate over batches of continuous z sections
        # The batches have a maximum size of `batch_size`
        # Batched iterations allows to utilize IO more efficiently
        first_z_idx = target_view.global_offset.z
        for source_file_batch in get_chunks(source_file_batches, batch_size):
            try:
                ref_time = time.time()
                logging.info(
                    "Cubing z={}-{}".format(
                        first_z_idx, first_z_idx + len(source_file_batch)
                    )
                )
                slices = []
                # Iterate over each z section in the batch
                for i, file_name in enumerate(source_file_batch):
                    z = first_z_idx + i
                    # Image shape will be (x, y, channel_count, z=1)
                    image = read_image_file(
                        file_name,
                        target_view.header.voxel_type,
                        z,
                        channel_index,
                        sample_index,
                    )

                    if not pad:
                        assert (
                            image.shape[0:2] == image_size
                        ), "Section z={} has the wrong dimensions: {} (expected {}). Consider using --pad.".format(
                            z, image.shape, image_size
                        )
                    slices.append(image)

                if pad:
                    x_max = max(_slice.shape[0] for _slice in slices)
                    y_max = max(_slice.shape[1] for _slice in slices)

                    slices = [
                        np.pad(
                            _slice,
                            mode="constant",
                            pad_width=[
                                (0, x_max - _slice.shape[0]),
                                (0, y_max - _slice.shape[1]),
                                (0, 0),
                                (0, 0),
                            ],
                        )
                        for _slice in slices
                    ]

                buffer = prepare_slices_for_wkw(slices, target_view.header.num_channels)
                if downsampling_needed:
                    buffer = downsample_unpadded_data(
                        buffer, target_mag, interpolation_mode
                    )
                buffer_z_offset = (
                    first_z_idx - target_view.global_offset.z
                ) // target_mag.z
                target_view.write(offset=(0, 0, buffer_z_offset), data=buffer)
                largest_value_in_chunk = max(largest_value_in_chunk, np.max(buffer))
                logging.debug(
                    "Cubing of z={}-{} took {:.8f}s".format(
                        first_z_idx,
                        first_z_idx + len(source_file_batch),
                        time.time() - ref_time,
                    )
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
    wkw_file_len: int,
    interpolation_mode_str: str,
    start_z: int,
    pad: bool,
    executor_args: Namespace,
    scale: Tuple[float, float, float] = (1, 1, 1),
) -> Layer:
    source_files = find_source_filenames(source_path)
    # we need to refresh the image readers because they are no longer stateless for performance reasons
    refresh_global_image_reader()

    # All images are assumed to have equal dimensions
    num_x, num_y = image_reader.read_dimensions(source_files[0])
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
        batch_size = BLOCK_LEN

    target_mag = Mag(target_mag_str)

    target_ds = Dataset.get_or_create(target_path, scale=scale)
    is_segmentation_layer = layer_name == "segmentation"

    if is_segmentation_layer:
        target_layer = target_ds.get_or_add_layer(
            layer_name,
            LayerCategories.SEGMENTATION_TYPE,
            dtype_per_channel=dtype,
            num_channels=num_output_channels,
            largest_segment_id=0,
        )
    else:
        target_layer = target_ds.get_or_add_layer(
            layer_name,
            LayerCategories.COLOR_TYPE,
            dtype_per_channel=dtype,
            num_channels=num_output_channels,
        )
    target_layer.bounding_box = BoundingBox(
        Vec3Int(0, 0, start_z) * target_mag,
        Vec3Int(num_x, num_y, num_z - start_z) * target_mag,
    )

    target_mag_view = target_layer.get_or_add_mag(
        target_mag, file_len=wkw_file_len, block_len=BLOCK_LEN
    )

    interpolation_mode = parse_interpolation_mode(interpolation_mode_str, target_layer.category)
    if target_mag != Mag(1):
        logging.info(
            f"Downsampling the cubed image to {target_mag} in memory with interpolation mode {interpolation_mode}."
        )

    logging.info("Found source files: count={} size={}x{}".format(num_z, num_x, num_y))

    with get_executor_for_args(executor_args) as executor:
        job_args = []
        # We iterate over all z sections
        for z in range(start_z, num_z, BLOCK_LEN):
            max_z = min(num_z, z + BLOCK_LEN)
            # Prepare source files array
            if len(source_files) > 1:
                source_files_array = source_files[z:max_z]
            else:
                source_files_array = source_files * (max_z - z)

            # Prepare job
            job_args.append(
                (
                    target_mag_view.get_view(
                        (0, 0, z),
                        (num_x, num_y, max_z - z),
                    ),
                    target_mag,
                    interpolation_mode,
                    source_files_array,
                    batch_size,
                    (num_x, num_y),
                    pad,
                    channel_index,
                    sample_index,
                )
            )

        largest_segment_id_per_chunk = wait_and_ensure_success(
            executor.map_to_futures(cubing_job, job_args)
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

    arg_dict = vars(args)

    cubing(
        args.source_path,
        args.target_path,
        args.layer_name,
        arg_dict.get("batch_size"),
        arg_dict.get("channel_index"),
        arg_dict.get("sample_index"),
        arg_dict.get("dtype"),
        args.target_mag,
        args.wkw_file_len,
        args.interpolation_mode,
        args.start_z,
        args.pad,
        args,
    )
