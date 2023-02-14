from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union, cast

import httpx

from ...client import Client
from ...models.dataset_list_response_200_item import DatasetListResponse200Item
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    is_active: Union[Unset, None, bool] = UNSET,
    is_unreported: Union[Unset, None, bool] = UNSET,
    is_editable: Union[Unset, None, bool] = UNSET,
    organization_name: Union[Unset, None, str] = UNSET,
    only_my_organization: Union[Unset, None, bool] = UNSET,
    uploader_id: Union[Unset, None, str] = UNSET,
    folder_id: Union[Unset, None, str] = UNSET,
    search_query: Union[Unset, None, str] = UNSET,
    limit: Union[Unset, None, int] = UNSET,
) -> Dict[str, Any]:
    url = "{}/api/datasets".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["isActive"] = is_active

    params["isUnreported"] = is_unreported

    params["isEditable"] = is_editable

    params["organizationName"] = organization_name

    params["onlyMyOrganization"] = only_my_organization

    params["uploaderId"] = uploader_id

    params["folderId"] = folder_id

    params["searchQuery"] = search_query

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "params": params,
    }


def _parse_response(
    *, response: httpx.Response
) -> Optional[Union[Any, List["DatasetListResponse200Item"]]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = DatasetListResponse200Item.from_dict(
                response_200_item_data
            )

            response_200.append(response_200_item)

        return response_200
    if response.status_code == HTTPStatus.BAD_REQUEST:
        response_400 = cast(Any, None)
        return response_400
    return None


def _build_response(
    *, response: httpx.Response
) -> Response[Union[Any, List["DatasetListResponse200Item"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    *,
    client: Client,
    is_active: Union[Unset, None, bool] = UNSET,
    is_unreported: Union[Unset, None, bool] = UNSET,
    is_editable: Union[Unset, None, bool] = UNSET,
    organization_name: Union[Unset, None, str] = UNSET,
    only_my_organization: Union[Unset, None, bool] = UNSET,
    uploader_id: Union[Unset, None, str] = UNSET,
    folder_id: Union[Unset, None, str] = UNSET,
    search_query: Union[Unset, None, str] = UNSET,
    limit: Union[Unset, None, int] = UNSET,
) -> Response[Union[Any, List["DatasetListResponse200Item"]]]:
    """List all accessible datasets.

    Args:
        is_active (Union[Unset, None, bool]):
        is_unreported (Union[Unset, None, bool]):
        is_editable (Union[Unset, None, bool]):
        organization_name (Union[Unset, None, str]):
        only_my_organization (Union[Unset, None, bool]):
        uploader_id (Union[Unset, None, str]):
        folder_id (Union[Unset, None, str]):
        search_query (Union[Unset, None, str]):
        limit (Union[Unset, None, int]):

    Returns:
        Response[Union[Any, List['DatasetListResponse200Item']]]
    """

    kwargs = _get_kwargs(
        client=client,
        is_active=is_active,
        is_unreported=is_unreported,
        is_editable=is_editable,
        organization_name=organization_name,
        only_my_organization=only_my_organization,
        uploader_id=uploader_id,
        folder_id=folder_id,
        search_query=search_query,
        limit=limit,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    *,
    client: Client,
    is_active: Union[Unset, None, bool] = UNSET,
    is_unreported: Union[Unset, None, bool] = UNSET,
    is_editable: Union[Unset, None, bool] = UNSET,
    organization_name: Union[Unset, None, str] = UNSET,
    only_my_organization: Union[Unset, None, bool] = UNSET,
    uploader_id: Union[Unset, None, str] = UNSET,
    folder_id: Union[Unset, None, str] = UNSET,
    search_query: Union[Unset, None, str] = UNSET,
    limit: Union[Unset, None, int] = UNSET,
) -> Optional[Union[Any, List["DatasetListResponse200Item"]]]:
    """List all accessible datasets.

    Args:
        is_active (Union[Unset, None, bool]):
        is_unreported (Union[Unset, None, bool]):
        is_editable (Union[Unset, None, bool]):
        organization_name (Union[Unset, None, str]):
        only_my_organization (Union[Unset, None, bool]):
        uploader_id (Union[Unset, None, str]):
        folder_id (Union[Unset, None, str]):
        search_query (Union[Unset, None, str]):
        limit (Union[Unset, None, int]):

    Returns:
        Response[Union[Any, List['DatasetListResponse200Item']]]
    """

    return sync_detailed(
        client=client,
        is_active=is_active,
        is_unreported=is_unreported,
        is_editable=is_editable,
        organization_name=organization_name,
        only_my_organization=only_my_organization,
        uploader_id=uploader_id,
        folder_id=folder_id,
        search_query=search_query,
        limit=limit,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    is_active: Union[Unset, None, bool] = UNSET,
    is_unreported: Union[Unset, None, bool] = UNSET,
    is_editable: Union[Unset, None, bool] = UNSET,
    organization_name: Union[Unset, None, str] = UNSET,
    only_my_organization: Union[Unset, None, bool] = UNSET,
    uploader_id: Union[Unset, None, str] = UNSET,
    folder_id: Union[Unset, None, str] = UNSET,
    search_query: Union[Unset, None, str] = UNSET,
    limit: Union[Unset, None, int] = UNSET,
) -> Response[Union[Any, List["DatasetListResponse200Item"]]]:
    """List all accessible datasets.

    Args:
        is_active (Union[Unset, None, bool]):
        is_unreported (Union[Unset, None, bool]):
        is_editable (Union[Unset, None, bool]):
        organization_name (Union[Unset, None, str]):
        only_my_organization (Union[Unset, None, bool]):
        uploader_id (Union[Unset, None, str]):
        folder_id (Union[Unset, None, str]):
        search_query (Union[Unset, None, str]):
        limit (Union[Unset, None, int]):

    Returns:
        Response[Union[Any, List['DatasetListResponse200Item']]]
    """

    kwargs = _get_kwargs(
        client=client,
        is_active=is_active,
        is_unreported=is_unreported,
        is_editable=is_editable,
        organization_name=organization_name,
        only_my_organization=only_my_organization,
        uploader_id=uploader_id,
        folder_id=folder_id,
        search_query=search_query,
        limit=limit,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)


async def asyncio(
    *,
    client: Client,
    is_active: Union[Unset, None, bool] = UNSET,
    is_unreported: Union[Unset, None, bool] = UNSET,
    is_editable: Union[Unset, None, bool] = UNSET,
    organization_name: Union[Unset, None, str] = UNSET,
    only_my_organization: Union[Unset, None, bool] = UNSET,
    uploader_id: Union[Unset, None, str] = UNSET,
    folder_id: Union[Unset, None, str] = UNSET,
    search_query: Union[Unset, None, str] = UNSET,
    limit: Union[Unset, None, int] = UNSET,
) -> Optional[Union[Any, List["DatasetListResponse200Item"]]]:
    """List all accessible datasets.

    Args:
        is_active (Union[Unset, None, bool]):
        is_unreported (Union[Unset, None, bool]):
        is_editable (Union[Unset, None, bool]):
        organization_name (Union[Unset, None, str]):
        only_my_organization (Union[Unset, None, bool]):
        uploader_id (Union[Unset, None, str]):
        folder_id (Union[Unset, None, str]):
        search_query (Union[Unset, None, str]):
        limit (Union[Unset, None, int]):

    Returns:
        Response[Union[Any, List['DatasetListResponse200Item']]]
    """

    return (
        await asyncio_detailed(
            client=client,
            is_active=is_active,
            is_unreported=is_unreported,
            is_editable=is_editable,
            organization_name=organization_name,
            only_my_organization=only_my_organization,
            uploader_id=uploader_id,
            folder_id=folder_id,
            search_query=search_query,
            limit=limit,
        )
    ).parsed
