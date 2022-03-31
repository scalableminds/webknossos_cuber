from argparse import ArgumentParser
from pathlib import Path

from webknossos import Dataset

from .utils import (
    add_data_format_flags,
    add_distribution_flags,
    add_verbose_flag,
    setup_logging,
)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Directory containing the datasource properties.", type=Path
    )

    parser.add_argument(
        "target_path", help="Output directory for the generated dataset.", type=Path
    )

    # TODO: Deprecated
    parser.add_argument(
        "--wkw_file_len", help="Target file length (default 32)", type=int, default=32
    )

    parser.add_argument(
        "--no_compression",
        help="Use compression, default false",
        type=bool,
        default=False,
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)
    add_data_format_flags(parser)

    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    Dataset.open(args.source_path).copy_dataset(
        args.target_path,
        data_format=args.data_format,
        chunk_size=args.chunk_size,
        chunks_per_shard=args.chunks_per_shard,
        compress=not args.no_compression,
        args=args,
    )
