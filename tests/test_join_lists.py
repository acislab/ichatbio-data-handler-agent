import json

import pytest
from ichatbio.agent_response import (
    ArtifactResponse,
)
from ichatbio.types import Artifact

from conftest import resource
from tools import join_lists


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url
    in ("https://artifact.test/list_one", "https://artifact.test/list_two")
)
@pytest.mark.asyncio
async def test_join_lists_by_index(context, messages, httpx_mock):
    list_one = json.loads(resource("list_of_idigbio_records.json"))
    httpx_mock.add_response(url="https://artifact.test/list_one", json=list_one)

    list_two = [{"new_field": "fee"}, {"new_field": "fi"}, {"new_field": "fo"}]
    httpx_mock.add_response(url="https://artifact.test/list_two", json=list_two)

    expected_list = [
        record_one | record_two for record_one, record_two in zip(list_one, list_two)
    ]

    await join_lists.join_lists(
        Artifact(
            local_id="#abcd",
            mimetype="application/json",
            description="A list of occurrence records",
            uris=["https://artifact.test/list_one"],
            metadata={},
        ),
        context,
        Artifact(
            local_id="#1234",
            mimetype="application/json",
            description="A list of extra data",
            uris=["https://artifact.test/list_two"],
            metadata={},
        ),
    )

    artifact_message = next(
        (m for m in messages if isinstance(m, ArtifactResponse)), None
    )
    assert artifact_message

    new_list = json.loads(artifact_message.content.decode("utf-8"))

    assert new_list == expected_list


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url
    in ("https://artifact.test/list_one", "https://artifact.test/list_two")
)
@pytest.mark.asyncio
async def test_dont_join_things_that_are_not_lists(context, messages, httpx_mock):
    list_one = json.loads('"this is not a list"')
    httpx_mock.add_response(url="https://artifact.test/list_one", json=list_one)

    list_two = [{"new_field": "fee"}, {"new_field": "fi"}, {"new_field": "fo"}]
    httpx_mock.add_response(url="https://artifact.test/list_two", json=list_two)

    await join_lists.join_lists(
        Artifact(
            local_id="#abcd",
            mimetype="application/json",
            description="A list of occurrence records",
            uris=["https://artifact.test/list_one"],
            metadata={},
        ),
        context,
        Artifact(
            local_id="#1234",
            mimetype="application/json",
            description="A list of extra data",
            uris=["https://artifact.test/list_two"],
            metadata={},
        ),
    )

    artifact_message = next(
        (m for m in messages if isinstance(m, ArtifactResponse)), None
    )
    assert artifact_message is None
