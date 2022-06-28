from typing import TYPE_CHECKING, Optional, Union

import attr

from webknossos.annotation.annotation import Annotation, AnnotationState, AnnotationType
from webknossos.utils import warn_deprecated

if TYPE_CHECKING:
    from webknossos.client._generated.models.annotation_info_response_200 import (
        AnnotationInfoResponse200,
    )
    from webknossos.client._generated.models.annotation_infos_by_task_id_response_200_item import (
        AnnotationInfosByTaskIdResponse200Item,
    )


@attr.frozen
class AnnotationInfo:
    """Data class containing information about a webKnossos annotation"""

    id: str
    owner_id: Optional[str]
    name: str
    description: str
    type: AnnotationType
    state: AnnotationState

    def download_annotation(self) -> Annotation:
        """Downloads and returns the annotation that is discribed by this AnnotationInfo object"""
        return Annotation.download(self.id, annotation_type=self.type)

    @classmethod
    def _from_generated_response(
        cls,
        response: Union[
            "AnnotationInfoResponse200", "AnnotationInfosByTaskIdResponse200Item"
        ],
    ) -> "AnnotationInfo":
        maybe_owner = getattr(response, "owner", None) or getattr(
            response, "user", None
        )
        owner_id = None
        if maybe_owner is not None:
            owner_id = maybe_owner.id
        return AnnotationInfo(
            id=response.id,
            owner_id=owner_id,
            name=response.name,
            description=response.description,
            type=AnnotationType(response.typ),
            state=AnnotationState(response.state),
        )

    @property
    def user_id(self) -> Optional[str]:
        warn_deprecated("user_id", "owner_id")
        return self.owner_id
