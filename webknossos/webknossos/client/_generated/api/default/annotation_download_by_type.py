from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import Client
from ...types import UNSET, Response, Unset


def _get_kwargs(
    typ: str,
    id: str,
    *,
    client: Client,
    skeleton_version: Union[Unset, None, int] = UNSET,
    volume_version: Union[Unset, None, int] = UNSET,
    skip_volume_data: Union[Unset, None, bool] = UNSET,
) -> Dict[str, Any]:
    url = "{}/api/annotations/{typ}/{id}/download".format(
        client.base_url, typ=typ, id=id
    )

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["skeletonVersion"] = skeleton_version

    params["volumeVersion"] = volume_version

    params["skipVolumeData"] = skip_volume_data

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "params": params,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Any]:
    if response.status_code == HTTPStatus.OK:
        return None
    if response.status_code == HTTPStatus.BAD_REQUEST:
        return None
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Client, response: httpx.Response) -> Response[Any]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    typ: str,
    id: str,
    *,
    client: Client,
    skeleton_version: Union[Unset, None, int] = UNSET,
    volume_version: Union[Unset, None, int] = UNSET,
    skip_volume_data: Union[Unset, None, bool] = UNSET,
) -> Response[Any]:
    """Download an annotation as NML/ZIP

    Args:
        typ (str):
        id (str):
        skeleton_version (Union[Unset, None, int]):
        volume_version (Union[Unset, None, int]):
        skip_volume_data (Union[Unset, None, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        typ=typ,
        id=id,
        client=client,
        skeleton_version=skeleton_version,
        volume_version=volume_version,
        skip_volume_data=skip_volume_data,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


async def asyncio_detailed(
    typ: str,
    id: str,
    *,
    client: Client,
    skeleton_version: Union[Unset, None, int] = UNSET,
    volume_version: Union[Unset, None, int] = UNSET,
    skip_volume_data: Union[Unset, None, bool] = UNSET,
) -> Response[Any]:
    """Download an annotation as NML/ZIP

    Args:
        typ (str):
        id (str):
        skeleton_version (Union[Unset, None, int]):
        volume_version (Union[Unset, None, int]):
        skip_volume_data (Union[Unset, None, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        typ=typ,
        id=id,
        client=client,
        skeleton_version=skeleton_version,
        volume_version=volume_version,
        skip_volume_data=skip_volume_data,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)
