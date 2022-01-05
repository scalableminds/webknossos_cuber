from pathlib import Path
from argparse import ArgumentParser
from typing import Optional
from os import environ

from webknossos import webknossos_context
from webknossos.client._defaults import DEFAULT_WEBKNOSSOS_URL
from wkcuber.api.dataset import Dataset

from .utils import add_verbose_flag, setup_logging


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
    url: str,
    token: Optional[str],
    jobs: int,
) -> None:
    with webknossos_context(url=url, token=token):
        Dataset.open(source_path).upload(jobs)


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    url = (
        args.url
        if args.url is not None
        else environ.get("WK_URL", DEFAULT_WEBKNOSSOS_URL)
    )
    token = args.token if args.token is not None else environ.get("WK_TOKEN", None)
    assert (
        token is not None
    ), f"An auth token needs to be supplied either through the --token command line arg or the WK_TOKEN environment variable. Retrieve your auth token on {webknossos_url}/auth/token."

    upload_dataset(args.source_path, url, token, args.jobs)
