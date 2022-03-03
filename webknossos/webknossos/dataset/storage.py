import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Type

import numpy as np
import wkw
import zarr
from numcodecs import Blosc

from webknossos.geometry import BoundingBox, Vec3Int, Vec3IntLike


class StorageArrayException(Exception):
    pass


@dataclass
class StorageArrayInfo:
    data_format: str
    num_channels: int
    voxel_type: np.dtype
    chunk_size: Vec3Int
    chunks_per_shard: Vec3Int
    compression_mode: bool = False

    @property
    def shard_size(self) -> Vec3Int:
        return self.chunk_size * self.chunks_per_shard


class StorageArray(ABC):
    data_format = ""

    _path: Path

    def __init__(self, path: Path):
        self._path = path

    @property
    @abstractmethod
    def info(self) -> StorageArrayInfo:
        pass

    @classmethod
    @abstractmethod
    def try_open(_cls, path: Path) -> Optional["StorageArray"]:
        for cls in (WKWStorageArray, ZarrStorageArray):
            array_maybe = cls.try_open(path)
            if array_maybe is not None:
                return array_maybe
        return None

    @classmethod
    @abstractmethod
    def create(cls, path: Path, storage_info: StorageArrayInfo) -> "StorageArray":
        pass

    @abstractmethod
    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        pass

    @abstractmethod
    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def list_bounding_boxes(self) -> Iterator[BoundingBox]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @staticmethod
    def get_class(data_format: str) -> Type["StorageArray"]:
        for cls in (WKWStorageArray, ZarrStorageArray):
            if cls.data_format == data_format:
                return cls
        raise ValueError(f"Data format `{data_format}` is invalid.")


class WKWStorageArray(StorageArray):
    data_format = "wkw"

    _cached_wkw_dataset: Optional[wkw.Dataset]

    def __init__(self, path: Path):
        super().__init__(path)
        self._cached_wkw_dataset = None

    @classmethod
    def try_open(cls, path: Path) -> Optional["WKWStorageArray"]:
        if (path / "header.wkw").is_file():
            return cls(path)
        return None

    @property
    def info(self) -> StorageArrayInfo:
        try:
            with wkw.Dataset.open(str(self._path)) as wkw_dataset:
                return StorageArrayInfo(
                    data_format=self.data_format,
                    num_channels=wkw_dataset.header.num_channels,
                    voxel_type=wkw_dataset.header.voxel_type,
                    compression_mode=wkw_dataset.header.block_type
                    != wkw.Header.BLOCK_TYPE_RAW,
                    chunk_size=Vec3Int.full(wkw_dataset.header.block_len),
                    chunks_per_shard=Vec3Int.full(
                        wkw_dataset.header.file_len,
                    ),
                )
        except wkw.wkw.WKWException as e:
            raise StorageArrayException(
                f"Exception while fetching storage info for {self._path}"
            ) from e

    @classmethod
    def create(cls, path: Path, storage_info: StorageArrayInfo) -> "WKWStorageArray":
        assert storage_info.data_format == cls.data_format
        assert storage_info.chunk_size[0] == storage_info.chunk_size[1]
        assert storage_info.chunk_size[0] == storage_info.chunk_size[2]
        assert storage_info.chunks_per_shard[0] == storage_info.chunks_per_shard[1]
        assert storage_info.chunks_per_shard[0] == storage_info.chunks_per_shard[2]
        try:
            wkw.Dataset.create(
                str(path),
                wkw.Header(
                    voxel_type=storage_info.voxel_type,
                    num_channels=storage_info.num_channels,
                    block_len=storage_info.chunk_size[0],
                    file_len=storage_info.chunks_per_shard[0],
                    block_type=(
                        wkw.Header.BLOCK_TYPE_LZ4HC
                        if storage_info.compression_mode
                        else wkw.Header.BLOCK_TYPE_RAW
                    ),
                ),
            ).close()
        except wkw.wkw.WKWException as e:
            raise StorageArrayException(f"Exception while creating array {path}") from e
        return WKWStorageArray(path)

    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        return self._wkw_dataset.read(offset, shape)

    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        self._wkw_dataset.write(offset, data)

    def _list_files(self) -> Iterator[Path]:
        return (
            Path(relpath(filename, self._path))
            for filename in self._wkw_dataset.list_files()
        )

    def list_bounding_boxes(self) -> Iterator[BoundingBox]:
        def _extract_num(s: str) -> int:
            match = re.search("[0-9]+", s)
            assert match is not None
            return int(match[0])

        def _extract_file_index(file_path: Path) -> Vec3Int:
            zyx_index = [_extract_num(el) for el in file_path.parts]
            return Vec3Int(zyx_index[2], zyx_index[1], zyx_index[0])

        shard_size = self.info.shard_size
        for file_path in self._list_files():
            cube_index = _extract_file_index(file_path)
            cube_offset = cube_index * shard_size

            yield BoundingBox(cube_offset, shard_size)

    def close(self) -> None:
        if self._cached_wkw_dataset is not None:
            self._cached_wkw_dataset.close()
            self._cached_wkw_dataset = None

    @property
    def _wkw_dataset(self) -> wkw.Dataset:
        if self._cached_wkw_dataset is None:
            self._cached_wkw_dataset = wkw.Dataset.open(
                str(self._path)
            )  # No need to pass the header to the wkw.Dataset
        return self._cached_wkw_dataset

    @_wkw_dataset.deleter
    def _wkw_dataset(self) -> None:
        self.close()

    def __del__(self) -> None:
        del self._cached_wkw_dataset

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        del d["_cached_wkw_dataset"]
        return d

    def __setstate__(self, d: Dict[str, Any]) -> None:
        d["_cached_wkw_dataset"] = None
        self.__dict__ = d


class ZarrStorageArray(StorageArray):
    data_format = "zarr"

    @classmethod
    def try_open(cls, path: Path) -> Optional["ZarrStorageArray"]:
        if (path / ".zarray").is_file():
            return cls(path)
        return None

    @property
    def info(self) -> StorageArrayInfo:
        zarray = zarr.open_array(self._path)
        return StorageArrayInfo(
            data_format=self.data_format,
            num_channels=zarray.shape[0],
            voxel_type=zarray.dtype,
            compression_mode=zarray.compressor is not None,
            chunk_size=Vec3Int(*zarray.chunks[1:4]) or Vec3Int.full(1),
            chunks_per_shard=Vec3Int.full(1),
        )

    @classmethod
    def create(cls, path: Path, storage_info: StorageArrayInfo) -> "ZarrStorageArray":
        assert storage_info.data_format == cls.data_format
        assert storage_info.chunks_per_shard == Vec3Int.full(1)
        zarr.create(
            shape=(storage_info.num_channels, 0, 0, 0),
            chunks=storage_info.chunk_size.to_tuple(),
            dtype=storage_info.voxel_type,
            compressor=(
                Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
                if storage_info.compression_mode
                else None
            ),
            store=path,
        )
        return ZarrStorageArray(path)

    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        offset = Vec3Int(offset)
        shape = Vec3Int(shape)
        return zarr.open(store=self._path)[
            :,
            offset.x : (offset.x + shape.x),
            offset.y : (offset.y + shape.y),
            offset.z : (offset.z + shape.z),
        ]

    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        offset = Vec3Int(offset)
        zarray = zarr.open(store=self._path)
        if data.ndim == 3:
            data = data.reshape((1,) + data.shape)
        assert data.ndim == 4
        new_shape = np.array((zarray.shape, offset.to_np() + data.shape[1:4]))
        zarray.resize(tuple(new_shape))
        zarray[
            :,
            offset.x : (offset.x + data.shape[1]),
            offset.y : (offset.y + data.shape[2]),
            offset.z : (offset.z + data.shape[3]),
        ] = data

    def list_bounding_boxes(self) -> Iterator[BoundingBox]:
        raise NotImplementedError()

    def close(self) -> None:
        pass
