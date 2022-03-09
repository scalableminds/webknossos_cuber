from typing import Any, Dict

import httpx

from ...client import Client
from ...types import Response


def _get_kwargs(
    organization_id: str,
    data_set_name: str,
    *,
    client: Client,
) -> Dict[str, Any]:
    url = f"{client.base_url}/api/datasets/{organization_id}/{data_set_name}/isValidNewName"

    headers: Dict[str, Any] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
    }


def _build_response(*, response: httpx.Response) -> Response[Any]:
    return Response(
        status_code=response.status_code,
        content=response.content,
        headers=response.headers,
        parsed=None,
    )


def sync_detailed(
    organization_id: str,
    data_set_name: str,
    *,
    client: Client,
) -> Response[Any]:
    kwargs = _get_kwargs(
        organization_id=organization_id,
        data_set_name=data_set_name,
        client=client,
    )

    response = httpx.get(
        **kwargs,
    )

    return _build_response(response=response)


async def asyncio_detailed(
    organization_id: str,
    data_set_name: str,
    *,
    client: Client,
) -> Response[Any]:
    kwargs = _get_kwargs(
        organization_id=organization_id,
        data_set_name=data_set_name,
        client=client,
    )

    async with httpx.AsyncClient() as _client:
        response = await _client.get(**kwargs)

    return _build_response(response=response)
