from typing import Tuple
from argparse import ArgumentParser
from pathlib import Path
from wkcuber.utils import (
    get_executor_for_args,
    named_partial,
    add_distribution_flags,
    add_scale_flag,
)
from wkcuber.api import Dataset, View


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Copies a given dataset layer in mag1 while moving all slices from z+1 to z. Can be used to fix off-by-one errors."
    )

    parser.add_argument("source_path", help="Path to input WKW dataset", type=Path)

    parser.add_argument(
        "target_path",
        help="WKW dataset with which to compare the input dataset.",
        type=Path,
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the layer to compare (if not provided, all layers are compared)",
        default=None,
    )

    add_distribution_flags(parser)
    add_scale_flag(parser)

    return parser


def move_by_one(
    src_mag,
    dst_mag,
    args: Tuple[View, int],
) -> None:
    chunk_view, i = args
    size = chunk_view.size
    dst_offset = chunk_view.global_offset

    src_offset = (
        dst_offset[0],
        dst_offset[1],
        dst_offset[2] + 1,
    )

    data = src_mag.read(src_offset, size)
    chunk_view.write(data)


def main():

    args = create_parser().parse_args()

    src_dataset = Dataset(args.source_path)
    src_layer = src_dataset.get_layer(args.layer_name)
    src_mag = src_layer.get_mag("1")

    dst_dataset = Dataset.get_or_create(args.target_path, args.scale)
    dst_layer = dst_dataset.add_layer(args.layer_name, "color")
    dst_layer.bounding_box = src_layer.bounding_box

    dst_mag = dst_layer.get_or_add_mag("1")

    dst_view = dst_mag.get_view()

    with get_executor_for_args(args) as executor:
        func = named_partial(move_by_one, src_mag, dst_mag)
        dst_view.for_each_chunk(
            func,
            chunk_size=dst_mag._get_file_dimensions(),
            executor=executor,
        )


if __name__ == "__main__":
    main()
