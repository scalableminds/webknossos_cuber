from http import HTTPStatus
from typing import Any, Dict

import httpx

from ...client import Client
from ...types import UNSET, Response


def _get_kwargs(
    typ: str,
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Dict[str, Any]:
    url = "{}/api/annotations/{typ}/{id}/info".format(client.base_url, typ=typ, id=id)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["timestamp"] = timestamp

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "params": params,
    }


def _build_response(*, response: httpx.Response) -> Response[Any]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=None,
    )


def sync_detailed(
    typ: str,
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Response[Any]:
    """Information about an annotation, supplying the type explicitly

    Args:
        typ (str):
        id (str):
        timestamp (int):

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        typ=typ,
        id=id,
        client=client,
        timestamp=timestamp,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    typ: str,
    id: str,
    *,
    client: Client,
    timestamp: int,
) -> Response[Any]:
    """Information about an annotation, supplying the type explicitly

    Args:
        typ (str):
        id (str):
        timestamp (int):

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        typ=typ,
        id=id,
        client=client,
        timestamp=timestamp,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)
