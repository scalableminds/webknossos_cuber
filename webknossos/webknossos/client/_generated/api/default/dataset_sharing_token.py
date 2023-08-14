from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import Client
from ...models.dataset_sharing_token_response_200 import DatasetSharingTokenResponse200
from ...types import Response


def _get_kwargs(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/api/datasets/{organizationName}/{dataSetName}/sharingToken".format(
        client.base_url, organizationName=organization_name, dataSetName=data_set_name
    )

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
    }


def _parse_response(
    *, client: Client, response: httpx.Response
) -> Optional[Union[Any, DatasetSharingTokenResponse200]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = DatasetSharingTokenResponse200.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.BAD_REQUEST:
        response_400 = cast(Any, None)
        return response_400
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Client, response: httpx.Response
) -> Response[Union[Any, DatasetSharingTokenResponse200]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
) -> Response[Union[Any, DatasetSharingTokenResponse200]]:
    """Sharing token of a dataset

    Args:
        organization_name (str):
        data_set_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, DatasetSharingTokenResponse200]]
    """

    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
) -> Optional[Union[Any, DatasetSharingTokenResponse200]]:
    """Sharing token of a dataset

    Args:
        organization_name (str):
        data_set_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, DatasetSharingTokenResponse200]
    """

    return sync_detailed(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
    ).parsed


async def asyncio_detailed(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
) -> Response[Union[Any, DatasetSharingTokenResponse200]]:
    """Sharing token of a dataset

    Args:
        organization_name (str):
        data_set_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, DatasetSharingTokenResponse200]]
    """

    kwargs = _get_kwargs(
        organization_name=organization_name,
        data_set_name=data_set_name,
        client=client,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    organization_name: str,
    data_set_name: str,
    *,
    client: Client,
) -> Optional[Union[Any, DatasetSharingTokenResponse200]]:
    """Sharing token of a dataset

    Args:
        organization_name (str):
        data_set_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, DatasetSharingTokenResponse200]
    """

    return (
        await asyncio_detailed(
            organization_name=organization_name,
            data_set_name=data_set_name,
            client=client,
        )
    ).parsed
