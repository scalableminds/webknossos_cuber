from functools import lru_cache
from pathlib import Path
from time import gmtime, strftime
from typing import Iterator, Optional, Tuple
from uuid import uuid4

import httpx
from rich.progress import Progress

from webknossos.client._generated.api.default import datastore_list
from webknossos.client._resumable import Resumable
from webknossos.client.context import _get_context, _WebknossosContext
from webknossos.dataset import Dataset


@lru_cache(maxsize=None)
def _cached_get_upload_datastore(context: _WebknossosContext) -> str:
    datastores = datastore_list.sync(client=context.generated_auth_client)
    assert datastores is not None
    for datastore in datastores:
        if datastore.allows_upload:
            assert isinstance(datastore.url, str)
            return datastore.url
    raise ValueError("No datastore found where datasets can be uploaded.")


def _walk_file(path: Path, base_path: Path) -> Iterator[Tuple[Path, Path, int]]:
    if path.is_dir():
        yield from _walk(path, base_path)
        return
    elif path.is_symlink():
        yield from _walk_file(path.resolve(), base_path)
        return
    yield (path.resolve(), path.relative_to(base_path), path.stat().st_size)


def _walk(
    path: Path, base_path: Optional[Path] = None
) -> Iterator[Tuple[Path, Path, int]]:
    if base_path is None:
        base_path = path
    for p in Path(path).iterdir():
        yield from _walk_file(p, base_path)


def upload_dataset(dataset: Dataset) -> str:
    context = _get_context()
    file_infos = list(_walk(dataset.path))
    total_file_size = sum(size for _, _, size in file_infos)
    # replicates https://github.com/scalableminds/webknossos/blob/master/frontend/javascripts/admin/dataset/dataset_upload_view.js
    time_str = strftime("%Y-%m-%dT%H-%M-%S", gmtime())
    upload_id = f"{time_str}__{uuid4()}"
    datastore_token = context.datastore_token
    datastore_url = _cached_get_upload_datastore(context)
    for _ in range(5):
        try:
            httpx.post(
                f"{datastore_url}/data/datasets/reserveUpload?token={datastore_token}",
                params={"token": datastore_token},
                json={
                    "uploadId": upload_id,
                    "organization": context.organization,
                    "name": dataset.name,
                    "totalFileCount": len(file_infos),
                    "initialTeams": [],
                },
                timeout=60,
            ).raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            http_error = e
    else:
        raise http_error
    with Progress() as progress:
        with Resumable(
            f"{datastore_url}/data/datasets?token={datastore_token}",
            simultaneous_uploads=1,
            query={
                "owningOrganization": context.organization,
                "name": dataset.name,
                "totalFileCount": 1,
            },
            chunk_size=100 * 1024 * 1024,  # 100 MiB
            generate_unique_identifier=lambda _, relative_path: f"{upload_id}/{relative_path}",
            test_chunks=False,
            permanent_errors=[400, 403, 404, 409, 415, 500, 501],
            client=httpx.Client(timeout=None),
        ) as session:
            progress_task = progress.add_task("Dataset Upload", total=total_file_size)
            for file_path, relative_path, _ in file_infos:
                resumable_file = session.add_file(file_path, relative_path)
                resumable_file.chunk_completed.register(
                    lambda chunk: progress.advance(progress_task, chunk.size)
                )
    for _ in range(5):
        try:
            httpx.post(
                f"{datastore_url}/data/datasets/finishUpload?token={datastore_token}",
                params={"token": datastore_token},
                json={
                    "uploadId": upload_id,
                    "organization": context.organization,
                    "name": dataset.name,
                    "needsConversion": False,
                },
                timeout=None,
            ).raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            http_error = e
    else:
        raise http_error

    return f"{context.url}/datasets/{context.organization}/{dataset.name}/view"
