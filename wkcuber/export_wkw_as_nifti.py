import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Dict

import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import zoom

from wkcuber.api.Dataset import WKDataset
from wkcuber.mag import Mag
from wkcuber.metadata import read_metadata_for_layer
from wkcuber.utils import (
    add_verbose_flag,
    add_distribution_flags,
    add_batch_size_flag,
    parse_bounding_box,
)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "--source_path", "-s", help="Directory containing the wkw file.", required=True
    )

    parser.add_argument(
        "--destination_path",
        "-d",
        help="Output directory for the generated tiff files.",
        required=True,
    )

    parser.add_argument("--name", "-n", help="Name of the tiffs", default="")

    parser.add_argument(
        "--bbox",
        help="The BoundingBox of which the tiff stack should be generated."
        "The input format is x,y,z,width,height,depth."
        "(By default, data for the full bounding box of the dataset is generated)",
        default=None,
        type=parse_bounding_box,
    )

    parser.add_argument(
        "--mag", "-m", help="The magnification that should be read", default=1
    )

    parser.add_argument(
        "--downsample", help="Downsample each tiff image", default=1, type=int
    )

    add_batch_size_flag(parser)

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def export_layer_to_nifti(
    wkw_file_path: Path,
    bbox: Dict,
    mag: Mag,
    layer_name: str,
    destination_path: Path,
    name: str,
) -> None:
    wk_ds = WKDataset(wkw_file_path)
    layer = wk_ds.get_layer(layer_name)
    mag_layer = layer.get_mag(mag)
    data = mag_layer.read(bbox["topleft"], bbox["size"])
    data = data.transpose(1, 2, 3, 0)
    # Problem: different dtypes (uint8 and uint32), implicit cast
    logging.info(f"Shape with layer {data.shape}")

    data = np.array(data)
    img = nib.Nifti1Image(data, np.eye(4))

    destination_file = str(destination_path.joinpath(name + ".nii"))

    logging.info(f"Writing to {destination_file} with shape {data.shape}")

    nib.save(img, destination_file)


def export_nifti(
    wkw_file_path: Path,
    bbox: Dict,
    mag: Mag,
    destination_path: Path,
    name: str,
):
    wk_ds = WKDataset(wkw_file_path)
    layers = wk_ds.layers

    for layer_name in layers:
        export_layer_to_nifti(
            wkw_file_path,
            bbox,
            mag,
            layer_name,
            destination_path,
            name + layer_name,
        )


def export_wkw_as_nifti(args: Namespace) -> None:
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.bbox is None:
        _, _, bbox_dim, origin = read_metadata_for_layer(
            args.source_path, args.layer_name
        )
        bbox = {"topleft": origin, "size": bbox_dim}
    else:
        bbox = {"topleft": list(args.bbox.topleft), "size": list(args.bbox.size)}

    logging.info(f"Starting tiff export for bounding box: {bbox}")

    export_nifti(
        wkw_file_path=args.source_path,
        bbox=bbox,
        mag=Mag(args.mag),
        destination_path=Path(args.destination_path),
        name=args.name,
    )


def run(args_list: List) -> None:
    arguments = create_parser().parse_args(args_list)
    export_wkw_as_nifti(arguments)


if __name__ == "__main__":
    arguments = create_parser().parse_args()
    export_wkw_as_nifti(arguments)
