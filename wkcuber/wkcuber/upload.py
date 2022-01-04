from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Optional
from os import environ

from webknossos import webknossos_context
from wkcuber.api.dataset import Dataset
from .mag import Mag

from .utils import add_verbose_flag, setup_logging
from typing import List


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Directory containing the source WKW dataset.", type=Path
    )

    parser.add_argument(
        "--token",
        help="Auth token of the user on webKnossos",
        default=None,
    )

    parser.add_argument(
        "--url", help="Base url of the webKnossos instance", default=None
    )

    parser.add_argument(
        "--jobs",
        "-j",
        default=5,
        type=int,
        help="Number of simultaneous upload processes.",
    )

    add_verbose_flag(parser)

    return parser


def upload_dataset(
    source_path: Path,
    token: Optional[str],
    url: str,
    jobs: int,
    args: Optional[Namespace] = None,
) -> None:
    with webknossos_context(url=url, token=token):
        Dataset.open(source_path).upload(jobs)


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    url = (
        args.url
        if args.url is not None
        else environ.get("WK_URL", "https://webknossos.org")
    )
    token = args.token if args.token is not None else environ.get("WK_TOKEN", None)
    assert (
        token is not None
    ), "An auth token needs to be supplied either through the --token command line arg or the WK_TOKEN environment variable."

    upload_dataset(args.source_path, url, token, args.jobs, args.mag, args)
