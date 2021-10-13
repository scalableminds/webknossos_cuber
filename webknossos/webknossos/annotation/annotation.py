import re
from enum import Enum, unique
from os import PathLike
from pathlib import Path
from typing import IO, List, NamedTuple, Optional, Union
from zipfile import ZipFile

from attr import dataclass
from boltons.cacheutils import cachedproperty

from webknossos.client.context import get_context
from webknossos.dataset import Dataset, Layer
from webknossos.skeleton import Skeleton, open_nml


class _ZipPath(NamedTuple):
    zipfile: ZipFile
    path: str

    def open(
        self, mode: str = "r", *, pwd: Optional[bytes] = None, force_zip64: bool = True
    ) -> IO[bytes]:
        assert "b" in mode, "Opening a ZipFile currently only supports binary mode"
        zip_mode = mode[0]
        return self.zipfile.open(
            self.path, mode=zip_mode, pwd=pwd, force_zip64=force_zip64
        )


@dataclass
class Annotation:
    file: Union[str, PathLike, IO[bytes]]

    @cachedproperty
    def _zipfile(self) -> ZipFile:
        return ZipFile(self.file)

    @cachedproperty
    def _filelist(self) -> List[str]:
        return [i.filename for i in self._zipfile.filelist]

    @cachedproperty
    def skeleton(self) -> Skeleton:
        nml_files = [i for i in self._filelist if i.endswith(".nml")]
        assert len(nml_files) == 1
        return open_nml(_ZipPath(self._zipfile, nml_files[0]))

    @cachedproperty
    def dataset_name(self) -> str:
        return self.skeleton.name

    def save_volume_annotation(
        self, dataset: Dataset, layer_name: str = "volume_annotation"
    ) -> Layer:
        assert "data.zip" in self._filelist
        with self._zipfile.open("data.zip") as f:
            ZipFile(f).extractall(dataset.path / layer_name)
        return dataset.add_layer_for_existing_files(
            layer_name, category="segmentation", largest_segment_id=0
        )


@unique
class AnnotationType(Enum):
    TASK = "Task"
    EXPLORATIONAL = "Explorational"
    COMPOUND_TASK = "CompoundTask"
    COMPOUND_PROJECT = "CompoundProject"
    COMPOUND_TASK_TYPE = "CompoundTaskType"


annotation_url_regex = re.compile(
    fr"(https?://.*)/annotations/({'|'.join(i.value for i in AnnotationType.__members__.values())})/([0-9A-Fa-f]*)"
)


def open_annotation(annotation_path: Union[str, PathLike]) -> "Annotation":
    if Path(annotation_path).exists():
        return Annotation(annotation_path)
    else:
        assert isinstance(
            annotation_path, str
        ), f"Called open_annotation with a path-like, but {annotation_path} does not exist."
        match = re.match(annotation_url_regex, annotation_path)
        assert (
            match is not None
        ), "open_annotation() must be called with a path or an annotation url, e.g. https://webknossos.org/annotations/Explorational/6114d9410100009f0096c640"
        webknossos_url, annotation_type_str, annotation_id = match.groups()
        annotation_type = AnnotationType(annotation_type_str)
        assert webknossos_url == get_context().url
        from webknossos.client.download_annotation import download_annotation

        return download_annotation(annotation_type, annotation_id)
