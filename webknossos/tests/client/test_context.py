import pytest

from webknossos.client.context import (
    _get_context,
    _WebknossosContext,
    webknossos_context,
)


@pytest.fixture
def env_context() -> _WebknossosContext:
    return _get_context()


@pytest.mark.use_proxay
def test_user_organization(env_context: _WebknossosContext) -> None:
    assert env_context.organization_id == "Organization_X"


@pytest.mark.use_proxay
def test_trailing_slash_in_url(env_context: _WebknossosContext) -> None:
    with webknossos_context(url=env_context.url + "/"):
        assert env_context.url == _get_context().url
