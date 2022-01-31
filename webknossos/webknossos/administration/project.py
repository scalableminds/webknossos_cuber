from typing import TYPE_CHECKING, List, Union

import attr

from webknossos.administration.user import User
from webknossos.client._generated.api.default import (
    project_info_by_id,
    project_info_by_name,
    task_infos_by_project_id,
)
from webknossos.client.context import _get_generated_client

if TYPE_CHECKING:
    from webknossos.administration import Task
    from webknossos.client._generated.models.project_info_by_id_response_200 import (
        ProjectInfoByIdResponse200,
    )
    from webknossos.client._generated.models.project_info_by_name_response_200 import (
        ProjectInfoByNameResponse200,
    )


@attr.frozen
class Project:
    project_id: str
    name: str
    team_id: str
    team_name: str
    owner_id: str
    priority: int
    paused: bool
    expected_time: int

    @classmethod
    def get_by_id(
        cls, project_id: str
    ) -> "Project":  # pylint: disable=redefined-builtin
        """Returns the project specified by the passed id if your token authorizes you to see it."""
        client = _get_generated_client(enforce_auth=True)
        response = project_info_by_id.sync(project_id, client=client)
        assert response is not None, "Could not fetch project by id."
        return cls._from_generated_response(response)

    @classmethod
    def get_by_name(cls, name: str) -> "Project":
        """Returns the user specified by the passed name if your token authorizes you to see it."""
        client = _get_generated_client(enforce_auth=True)
        response = project_info_by_name.sync(name, client=client)
        assert response is not None, "Could not fetch project by name."
        return cls._from_generated_response(response)

    def get_tasks(self, fetch_all: bool = True) -> List["Task"]:
        from webknossos.administration import Task

        pagination_limit = 1000
        pagination_page = 0

        client = _get_generated_client(enforce_auth=True)
        response_raw = task_infos_by_project_id.sync_detailed(
            self.project_id,
            limit=pagination_limit,
            page_number=pagination_page,
            include_total_count=True,
            client=client,
        )
        total_count_raw = response_raw.headers.get("X-Total-Count")
        assert total_count_raw is not None, "X-Total-Count header missing from response"
        total_count = int(total_count_raw)
        if total_count > pagination_limit and not fetch_all:
            print(
                f"Fetched only {pagination_limit} of {total_count} tasks. Pass fetch_all=True to fetch all tasks iteratively (may be slow!)"
            )
        response = response_raw.parsed
        assert response is not None, "Could not fetch task infos by project id."
        return [Task._from_generated_response(t) for t in response]

    def get_owner(self) -> User:
        return User.get_by_id(self.owner_id)

    @classmethod
    def _from_generated_response(
        cls,
        response: Union["ProjectInfoByIdResponse200", "ProjectInfoByNameResponse200"],
    ) -> "Project":
        return cls(
            response.id,
            response.name,
            response.team,
            response.team_name,
            response.owner.id,
            response.priority,
            bool(response.paused),
            response.expected_time,
        )
