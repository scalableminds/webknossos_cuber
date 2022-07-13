import shutil
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import IO, Dict, List, Tuple

import cairosvg
import numpy as np
import webknossos as wk
from attr import evolve
from PIL import Image
from svgwrite import Drawing, px
from svgwrite.shapes import Polygon
from webknossos.utils import (
    add_verbose_flag,
    get_executor_for_args,
    setup_logging,
    setup_warnings,
    wait_and_ensure_success,
)

from ._internal.utils import add_distribution_flags

logger = getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None


@dataclass
class Model:
    bounding_box: wk.BoundingBox
    voxel_size: Tuple[float, float, float]
    objects: List["Object"]


@dataclass
class Object:
    id: int
    color: Tuple[float, float, float, float]
    contours: List["Contour"]

    def grouped_contours(self) -> Dict[int, List["Contour"]]:
        output = defaultdict(list)
        for contour in self.contours:
            output[contour.z] == contour

    def draw_svg_paths(self, z: int, drawing: Drawing, height: int) -> List[Polygon]:
        z_contours = [c for c in self.contours if c.z == z]
        for contour in z_contours:
            drawing.add(
                drawing.polygon(
                    points=[(p[0], height - p[1]) for p in contour.points],
                    fill=f"rgb({(self.id + 1) % 256},0,0)",
                    shape_rendering="crispEdges",
                )
            )


@dataclass
class Contour:
    id: int
    z: int
    flags: int
    points: np.ndarray


def parse_line(line: str) -> List[str]:
    parts = [part for part in line.rstrip().split(" ") if part != ""]
    if len(parts) == 0:
        return [""]
    return parts


def parse_model(f: IO) -> Model:
    bounding_box = wk.BoundingBox(wk.Vec3Int.zeros(), wk.Vec3Int.zeros())
    voxel_size = (1, 1, 1)
    object_count = 0
    objects = []

    for line in f:
        parts = parse_line(line)
        if parts[0] == "imod":
            object_count = int(parts[1])
        elif parts[0] == "max":
            bounding_box = evolve(
                bounding_box,
                size=wk.Vec3Int(int(parts[1]), int(parts[2]), int(parts[3])),
            )
        elif parts[0] == "offsets":
            bounding_box = evolve(
                bounding_box,
                topleft=wk.Vec3Int(int(parts[1]), int(parts[2]), int(parts[3])),
            )
        elif parts[0] == "scale":
            voxel_size = (
                voxel_size[0] * float(parts[1]),
                voxel_size[1] * float(parts[2]),
                voxel_size[2] * float(parts[3]),
            )
        elif parts[0] == "pixsize":
            voxel_size = (
                voxel_size[0] * float(parts[1]),
                voxel_size[1] * float(parts[1]),
                voxel_size[2] * float(parts[1]),
            )
        elif parts[0] == "units":
            if parts[1] == "um":
                voxel_size = (
                    voxel_size[0] * 1000,
                    voxel_size[1] * 1000,
                    voxel_size[2] * 1000,
                )
        elif parts[0] == "object":
            objects.append(parse_object(parts, f))
            if len(objects) == object_count:
                break

    assert (
        len(objects) == object_count
    ), f"Actual object count does not match expected object count: {len(objects)} != {object_count}"
    return Model(bounding_box=bounding_box, voxel_size=voxel_size, objects=objects)


def parse_object(parts: List[str], f: IO) -> Object:
    id = int(parts[1])
    color = (0, 0, 0, 0)
    contour_count = int(parts[2])
    contours = []

    for line in f:
        parts = parse_line(line)
        if parts[0] == "color":
            color = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        elif parts[0] == "contour":
            contours.append(parse_contour(parts, f))
        elif parts[0] == "contflags":
            contours[-1].flags = int(parts[1])
        elif parts[0] == "":
            break

    assert (
        len(contours) == contour_count
    ), f"Actual contour count does not match expected contour count: {len(contours)} != {contour_count}"
    return Object(id=id, color=color, contours=contours)


def parse_contour(parts: List[str], f: IO) -> Contour:
    id = int(parts[1])
    point_count = int(parts[3])
    points = np.zeros((point_count, 2), dtype=np.float32)
    z = -1

    for i in range(point_count):
        parts = parse_line(next(f))
        if z == -1:
            z = int(parts[2])
        else:
            assert z == int(
                parts[2]
            ), f"All points of a contour need to be in one section {z}"
        points[i, :] = (float(parts[0]), float(parts[1]))

    return Contour(id=id, z=z, flags=0, points=points)


def convert_slices(args: Tuple[wk.BoundingBox, Model, wk.MagView, Path]) -> None:
    (z_bbox, model, target_mag, target_path) = args

    for z in range(z_bbox.topleft.z, z_bbox.bottomright.z):
        svg_file_path = str(target_path / "temp" / f"{z:05}.svg")
        png_file_path = str(target_path / "temp" / f"{z:05}.png")
        drawing = Drawing(
            svg_file_path,
            size=(model.bounding_box.size.x, model.bounding_box.size.y),
        )
        for object in model.objects:
            object.draw_svg_paths(z, drawing, model.bounding_box.size.y)
        drawing.save()

        cairosvg.svg2png(url=svg_file_path, write_to=png_file_path)

    stack = np.zeros(z_bbox.size, dtype="uint32")
    for i, z in enumerate(range(z_bbox.topleft.z, z_bbox.bottomright.z)):
        png_file_path = str(target_path / "temp" / f"{z:05}.png")
        section_data = np.array(Image.open(png_file_path), dtype="uint8")
        section_data = section_data[:, :, 0]
        stack[:, :, i] = section_data.T

    target_mag.write(absolute_offset=z_bbox.topleft, data=stack)


def main(args: Namespace) -> None:

    with args.source_path.open("rt") as f:
        model = parse_model(f)

        logger.info("Done parsing")

        shutil.rmtree(args.target_path / "temp")
        shutil.rmtree(args.target_path / "wkw")
        (args.target_path / "temp").mkdir(exist_ok=True, parents=True)

        logger.info("Generating WKW dataset")
        dataset = wk.Dataset(args.target_path / "wkw", voxel_size=model.voxel_size)
        seg_layer = dataset.add_layer(
            "segmentation",
            wk.SEGMENTATION_CATEGORY,
            dtype_per_channel="uint32",
            largest_segment_id=max(obj.id for obj in model.objects) + 1,
        )
        seg_layer.bounding_box = model.bounding_box
        seg_mag = seg_layer.add_mag(1)

        with get_executor_for_args(args) as executor:
            job_args = [
                (z_bbox, model, seg_mag, args.target_path)
                for z_bbox in model.bounding_box.chunk(
                    (10000000000000, 10000000000000, 32), (32, 32, 32)
                )
            ]

            wait_and_ensure_success(
                executor.map_to_futures(
                    convert_slices,
                    job_args,
                ),
                progress_desc=f"Converting {model.bounding_box.size.z} slices in {len(job_args)} batches",
            )

        seg_mag.compress(args=args)
        seg_layer.downsample(args=args)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path",
        help="IMOD file (ascii format) containing the volume annotations",
        type=Path,
    )

    parser.add_argument(
        "target_path",
        help="Output directory for the generated dataset",
        type=Path,
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


if __name__ == "__main__":
    setup_warnings()
    args = create_parser().parse_args()
    setup_logging(args)
    main(args)
