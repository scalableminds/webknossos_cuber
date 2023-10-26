from typing import Dict, Optional, Tuple

from webknossos.client.apiclient.models import (
    ApiReserveUploadInformation,
    ApiUploadInformation,
)

from ._abstract_api_client import LONG_TIMEOUT_SECONDS, AbstractApiClient


class DatastoreApiClient(AbstractApiClient):
    def __init__(
        self,
        datastore_base_url: str,
        timeout_seconds: float,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(timeout_seconds, headers)
        self.datastore_base_url = datastore_base_url

    @property
    def url_prefix(self) -> str:
        return f"{self.datastore_base_url}/data"

    def dataset_finish_upload(
        self,
        upload_information: ApiUploadInformation,
        token: Optional[str],
        retry_count: int,
    ) -> None:
        route = "/datasets/finishUpload"
        return self._post_json(
            route,
            upload_information,
            query={"token": token},
            retry_count=retry_count,
            timeout_seconds=LONG_TIMEOUT_SECONDS,
        )

    def dataset_reserve_upload(
        self,
        reserve_upload_information: ApiReserveUploadInformation,
        token: Optional[str],
        retry_count: int,
    ) -> None:
        route = "/datasets/reserveUpload"
        return self._post_json(
            route,
            reserve_upload_information,
            query={"token": token},
            retry_count=retry_count,
        )

    def dataset_get_raw_data(self,
                             organization_name: str,
                             dataset_name: str,
                             data_layer_name: str,
                             mag: str,
                             token: Optional[str],
                             x: int,
                             y: int,
                             z: int,
                             width: int,
                             height: int,
                             depth: int) -> Tuple[bytes, str]:
        route = f"/datasets/{organization_name}/{dataset_name}/layers/{data_layer_name}/data"
        query = {
            "mag":mag,
            "x":x,
            "y":y,
            "z":z,
            "width":width,
            "height":height,
            "depth":depth,
            "token":token,
        }
        response = self._get(route, query)
        return response.content, response.headers.get("MISSING-BUCKETS")
