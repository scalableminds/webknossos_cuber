import logging
import math
from typing import Tuple, cast

import numpy as np
from argparse import ArgumentParser, Namespace
import os

from wkcuber.api.Dataset import WKDataset
from .mag import Mag
from .metadata import read_datasource_properties

from .utils import (
    add_verbose_flag,
    add_distribution_flags,
    add_interpolation_flag,
    add_isotropic_flag,
    setup_logging,
)


def calculate_virtual_scale_for_target_mag(
    target_mag: Mag,
) -> Tuple[float, float, float]:
    """
    This scale is not the actual scale of the dataset
    The virtual scale is used for downsample_mags_anisotropic.
    """
    max_target_value = max(list(target_mag.to_array()))
    scale_array = max_target_value / np.array(target_mag.to_array())
    return cast(Tuple[float, float, float], tuple(scale_array))


def calculate_default_max_mag_for_dataset(dataset_path: str, layer_name: str) -> Mag:
    ds = WKDataset(dataset_path)
    ds_size = ds.properties.data_layers[layer_name].get_bounding_box_size()
    cube_length = (
        ds.properties.data_layers[layer_name].wkw_magnifications[0].cube_length
    )
    return calculate_default_max_mag(ds_size, cube_length)


def calculate_default_max_mag(
    dataset_size: Tuple[int, int, int], cube_length: int
) -> Mag:
    cubes_per_dim = -(-np.array(dataset_size) // np.array(cube_length))
    magnification = [
        int(math.pow(2, int(math.log(s, 2)))) for s in cubes_per_dim
    ]  # highest power of 2 smaller than cubes_per_dim
    return Mag(
        [min(max(m, 16), 512) for m in magnification]
    )  # at least 16 and at most 512


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("path", help="Directory containing the dataset.")

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation)",
        default="color",
    )

    parser.add_argument(
        "--from_mag",
        "--from",
        "-f",
        help="Resolution to base downsampling on",
        type=str,
        default="1",
    )

    # Either provide the maximum resolution to be downsampled OR a specific, anisotropic magnification.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--max",
        "-m",
        help="Max resolution to be downsampled. In case of anisotropic downsampling, the process is considered "
        "done when max(current_mag) >= max(max_mag) where max takes the largest dimension of the mag tuple "
        "x, y, z. For example, a maximum mag value of 8 (or 8-8-8) will stop the downsampling as soon as a "
        "magnification is produced for which one dimension is equal or larger than 8.",
        type=int,
        default=None,
    )

    group.add_argument(
        "--anisotropic_target_mag",
        help="Specify an explicit anisotropic target magnification (e.g., --anisotropic_target_mag 16-16-4)."
        "All magnifications until this target magnification will be created. Consider using --anisotropic "
        "instead which automatically creates multiple anisotropic magnifications depending "
        "on the dataset's scale",
        type=str,
    )

    parser.add_argument(
        "--buffer_cube_size",
        "-b",
        help="Size of buffered cube to be downsampled (i.e. buffer cube edge length)",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no_compress",
        help="Don't compress data during downsampling",
        default=False,
        action="store_true",
    )

    add_interpolation_flag(parser)
    add_verbose_flag(parser)
    add_isotropic_flag(parser)
    add_distribution_flags(parser)

    return parser


def downsample_mags(
    path: str,
    layer_name: str = None,
    from_mag: Mag = None,
    max_mag: Mag = Mag(32),
    interpolation_mode: str = "default",
    buffer_edge_len: int = None,
    compress: bool = True,
    args: Namespace = None,
    anisotropic: bool = True,
) -> None:
    assert layer_name and from_mag or not layer_name and not from_mag, (
        "You provided only one of the following "
        "parameters: layer_name, from_mag but both "
        "need to be set or none. If you don't provide "
        "the parameters you need to provide the path "
        "argument with the mag and layer to downsample"
        " (e.g dataset/color/1)."
    )
    scale = getattr(args, "scale", None) if args else None
    if not layer_name or not from_mag:
        layer_name = os.path.basename(os.path.dirname(path))
        from_mag = Mag(os.path.basename(path))
        path = os.path.dirname(os.path.dirname(path))

    WKDataset(path).get_layer(layer_name).downsample(
        from_mag=from_mag,
        max_mag=max_mag,
        interpolation_mode=interpolation_mode,
        compress=compress,
        anisotropic=anisotropic,
        scale=scale,
        buffer_edge_len=buffer_edge_len,
        args=args,
    )


def downsample_mags_isotropic(
    path: str,
    layer_name: str,
    from_mag: Mag,
    max_mag: Mag,
    interpolation_mode: str,
    compress: bool,
    buffer_edge_len: int = None,
    args: Namespace = None,
) -> None:

    WKDataset(path).get_layer(layer_name).downsample(
        from_mag=from_mag,
        max_mag=max_mag,
        interpolation_mode=interpolation_mode,
        compress=compress,
        anisotropic=False,
        scale=None,
        buffer_edge_len=buffer_edge_len,
        args=args,
    )


def downsample_mags_anisotropic(
    path: str,
    layer_name: str,
    from_mag: Mag,
    max_mag: Mag,
    scale: Tuple[float, float, float],
    interpolation_mode: str,
    compress: bool,
    buffer_edge_len: int = None,
    args: Namespace = None,
) -> None:
    WKDataset(path).get_layer(layer_name).downsample(
        from_mag=from_mag,
        max_mag=max_mag,
        interpolation_mode=interpolation_mode,
        compress=compress,
        anisotropic=True,
        scale=scale,
        buffer_edge_len=buffer_edge_len,
        args=args,
    )


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    from_mag = Mag(args.from_mag)
    if args.max is None:
        max_mag = calculate_default_max_mag_for_dataset(args.path, args.layer_name)
    else:
        max_mag = Mag(args.max)
    if args.anisotropic_target_mag:
        anisotropic_target_mag = Mag(args.anisotropic_target_mag)

        scale = calculate_virtual_scale_for_target_mag(anisotropic_target_mag)

        downsample_mags_anisotropic(
            args.path,
            args.layer_name,
            from_mag,
            anisotropic_target_mag,
            scale,
            args.interpolation_mode,
            not args.no_compress,
            args.buffer_cube_size,
            args,
        )
    elif not args.isotropic:
        try:
            scale = read_datasource_properties(args.path)["scale"]
        except Exception as exc:
            logging.error(
                "Could not determine scale which is necessary "
                "to find target magnifications for anisotropic downsampling. "
                "Does the provided dataset have a datasource-properties.json file?"
            )
            raise exc

        downsample_mags_anisotropic(
            args.path,
            args.layer_name,
            from_mag,
            max_mag,
            scale,
            args.interpolation_mode,
            not args.no_compress,
            args=args,
        )
    else:
        downsample_mags_isotropic(
            args.path,
            args.layer_name,
            from_mag,
            max_mag,
            args.interpolation_mode,
            not args.no_compress,
            args.buffer_cube_size,
            args,
        )
