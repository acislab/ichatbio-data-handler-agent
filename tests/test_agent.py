import json

import ichatbio.agent_response
import pytest
import pytest_asyncio
from ichatbio.agent_response import (
    ArtifactResponse,
)
from ichatbio.agent_response import DirectResponse
from ichatbio.types import Artifact

import agent
import test_resources
from agent import DataHandlerAgent

OCCURRENCE_RECORDS = Artifact(
    local_id="#0000",
    mimetype="application/json",
    description="A list of occurrence records",
    uris=["https://artifact.test"],
    metadata={"source": "iDigBio"},
)


@pytest_asyncio.fixture()
def data_handler():
    return DataHandlerAgent()


@pytest.mark.asyncio
async def test_abort_without_appropriate_tool(data_handler, context, messages):
    await data_handler.run(
        context,
        "Draw a giraffe",
        "process_data",
        agent.EntrypointParameters(artifacts=[OCCURRENCE_RECORDS]),
    )

    assert len(messages) == 1
    assert isinstance(messages[0], DirectResponse)


@pytest.mark.httpx_mock(should_mock=lambda request: request.url == "https://artifact.test")
@pytest.mark.asyncio
async def test_abort_after_failed_query(data_handler, context, messages, httpx_mock):
    source_data = json.loads((test_resources.dir / "idigbio_records_search_result.json").read_text())
    httpx_mock.add_response(url="https://artifact.test", json=source_data)

    await data_handler.run(
        context,
        'Extract the "dwc.moonphases" field from these records',
        "process_data",
        agent.EntrypointParameters(artifacts=[OCCURRENCE_RECORDS]),
    )

    assert len(messages) > 1
    assert isinstance(messages[0], ichatbio.agent_response.ProcessBeginResponse)
    assert not any([isinstance(message, ArtifactResponse) for message in messages])


def test_system_message():
    system_message = agent.make_system_message(
        "Here they are:\n\n{artifacts}", [OCCURRENCE_RECORDS, OCCURRENCE_RECORDS]
    )

    expected = """\
Here they are:

- local_id: #0000
  description: A list of occurrence records
  uris: ['https://artifact.test']
  metadata: {'source': 'iDigBio'}

- local_id: #0000
  description: A list of occurrence records
  uris: ['https://artifact.test']
  metadata: {'source': 'iDigBio'}\
"""

    assert system_message == expected


def test_system_message_with_no_artifacts():
    system_message = agent.make_system_message("Any artifacts? {artifacts}", {})

    expected = "Any artifacts? NO AVAILABLE ARTIFACTS"

    assert system_message == expected
