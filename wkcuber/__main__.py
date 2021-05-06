from typing import List, Dict

from wkcuber import downsample_mags
from .compress import compress_mag_inplace
from .metadata import refresh_metadata
from .utils import add_isotropic_flag, setup_logging, add_sampling_mode_flag
from .mag import Mag
from .converter import (
    create_parser as create_conversion_parser,
    main as auto_detect_and_run_conversion,
)
from argparse import Namespace, ArgumentParser
from pathlib import Path


def detect_present_mags(target_path: Path) -> Dict[Path, List[Mag]]:
    layer_path_to_mags: Dict[Path, List[Mag]] = dict()
    layer_paths = list([p for p in target_path.iterdir() if p.is_dir()])
    for layer_p in layer_paths:
        layer_path_to_mags.setdefault(layer_p, list())
        mag_paths = list([p for p in layer_p.iterdir() if p.is_dir()])
        for mag_p in mag_paths:
            try:
                mag = Mag(mag_p.stem)
            except (AssertionError, ValueError) as _:
                continue
            layer_path_to_mags[layer_p].append(mag)

    return layer_path_to_mags


def create_parser() -> ArgumentParser:
    parser = create_conversion_parser()

    parser.add_argument(
        "--max_mag",
        "-m",
        help="Max resolution to be downsampled. Needs to be a power of 2.",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--no_compress",
        help="Don't compress this data",
        default=False,
        action="store_true",
    )

    parser.add_argument("--name", "-n", help="Name of the dataset", default=None)
    add_isotropic_flag(parser)
    add_sampling_mode_flag(parser)

    return parser


def main(args: Namespace) -> None:
    setup_logging(args)

    if args.isotropic is not None:
        raise DeprecationWarning(
            "The flag 'isotropic' is deprecated. Consider using 'sampling_mode' instead."
        )

    auto_detect_and_run_conversion(args)

    layer_path_to_mags: Dict[Path, List[Mag]] = detect_present_mags(args.target_path)

    if not args.no_compress:
        for (layer_path, mags) in layer_path_to_mags.items():
            layer_name = layer_path.stem
            for mag in mags:
                compress_mag_inplace(args.target_path, layer_name, mag, args)

    for (layer_path, mags) in layer_path_to_mags.items():
        layer_name = layer_path.stem
        mags.sort()
        downsample_mags(
            path=args.target_path,
            layer_name=layer_name,
            from_mag=mags[-1],
            max_mag=Mag(args.max_mag),
            interpolation_mode="default",
            compress=not args.no_compress,
            sampling_mode=args.sampling_mode,
            args=args,
        )

    refresh_metadata(args.target_path)


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    main(args)
