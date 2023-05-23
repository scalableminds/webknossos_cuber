"""This module takes care of upsampling WEBKNOSSOS datasets."""

from argparse import Namespace
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from typing_extensions import Annotated

from webknossos import Dataset, Mag, Vec3Int
from webknossos.dataset.sampling_modes import SamplingModes
from webknossos.utils import get_executor_for_args
from wkcuber.utils import DistributionStrategy, SamplingMode, parse_mag


def main(
    *,
    target: Annotated[
        Path,
        typer.Argument(help="Path to your WEBKNOSSOS dataset.", show_default=False),
    ],
    sampling_mode: Annotated[
        SamplingMode, typer.Option(help="The sampling mode to use.")
    ] = SamplingMode.ANISOTROPIC,
    from_mag: Annotated[
        Mag,
        typer.Option(
            help="Mag to start upsampling from.\
Should be number or minus seperated string (e.g. 2 or 2-2-2).",
            parser=parse_mag,
        ),
    ],
    layer_name: Annotated[
        Optional[str],
        typer.Option(
            help="Name of the layer that should be downsampled.", show_default=False
        ),
    ] = None,
    jobs: Annotated[
        int,
        typer.Option(
            help="Number of processes to be spawned.",
            rich_help_panel="Executor options",
        ),
    ] = cpu_count(),
    distribution_strategy: Annotated[
        DistributionStrategy,
        typer.Option(
            help="Strategy to distribute the task across CPUs or nodes.",
            rich_help_panel="Executor options",
        ),
    ] = DistributionStrategy.MULTIPROCESSING,
    job_resources: Annotated[
        Optional[str],
        typer.Option(
            help='Necessary when using slurm as distribution strategy. Should be a JSON string \
(e.g., --job_resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Upsample your WEBKNOSSOS dataset."""

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy,
        job_resources=job_resources,
    )
    dataset = Dataset.open(target)
    mode = SamplingModes.parse(sampling_mode.value)
    mag = Mag(Vec3Int.from_xyz(*from_mag))
    with get_executor_for_args(args=executor_args) as executor:
        if layer_name is None:
            upsample_all_layers(dataset, mode, mag, executor_args)
        else:
            layer = dataset.get_layer(layer_name)
            layer.upsample(from_mag=mag, sampling_mode=mode, executor=executor)

    rprint("[bold green]Done.[/bold green]")


def upsample_all_layers(
    dataset: Dataset, mode: SamplingModes, from_mag: Mag, executor_args: Namespace
) -> None:
    """Iterates over all layers and upsamples them."""

    for layer in dataset.layers.values():
        with get_executor_for_args(args=executor_args) as executor:
            layer.upsample(
                from_mag=from_mag,
                sampling_mode=mode,
                executor=executor,
            )
