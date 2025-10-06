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
from conftest import resource
from src.agent import DataHandlerAgent
from src.tools import process_data

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


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url == "https://artifact.test"
)
@pytest.mark.asyncio
async def test_extract_first_record(data_handler, context, messages, httpx_mock):
    source_data = json.loads(resource("idigbio_records_search_result.json"))
    httpx_mock.add_response(url="https://artifact.test", json=source_data)

    await data_handler.run(
        context,
        "Get the first record",
        "process_data",
        agent.EntrypointParameters(artifacts=[OCCURRENCE_RECORDS]),
    )

    artifact_message = next((m for m in messages if isinstance(m, ArtifactResponse)))
    assert artifact_message

    first_record = json.loads(artifact_message.content.decode("utf-8"))

    # If the record was put into a single-item list, pull it out
    if type(first_record) is list:
        assert len(first_record) == 1
        first_record = first_record[0]

    assert first_record == source_data["items"][0]


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


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url == "https://artifact.test"
)
@pytest.mark.asyncio
async def test_abort_after_failed_query(data_handler, context, messages, httpx_mock):
    source_data = json.loads(resource("idigbio_records_search_result.json"))
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
        {"#0000": OCCURRENCE_RECORDS, "#1111": OCCURRENCE_RECORDS}
    )

    expected = """\
You manipulate structured data using tools. You can access the following artifacts:

- local_id: #0000
  description: A list of occurrence records
  uris: ['https://artifact.test']
  metadata: {'source': 'iDigBio'}

- local_id: #0000
  description: A list of occurrence records
  uris: ['https://artifact.test']
  metadata: {'source': 'iDigBio'}

If you are unable to fulfill the user's request using your available tools, abort and explain why.\
"""

    assert system_message == expected


def test_system_message_with_no_artifacts():
    system_message = agent.make_system_message({})

    expected = """\
You manipulate structured data using tools. You can access the following artifacts:

NO AVAILABLE ARTIFACTS

If you are unable to fulfill the user's request using your available tools, abort and explain why.\
"""

    assert system_message == expected


@pytest.mark.asyncio
async def test__generate_jq_query():
    generation, result = await process_data._generate_and_run_jq_query(
        request="Get me the first bug",
        schema={"type": "string"},
        source_content=["cricket", "bumblebee"],
        source_artifact=Artifact(
            local_id="#0000",
            mimetype="application/json",
            description="Just some bugs",
            uris=["https://artifact.test"],
            metadata={},
        ),
    )

    assert result == ["cricket"]


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url
    in ("https://artifact.test/list_one", "https://artifact.test/list_two")
)
@pytest.mark.asyncio
async def test_join_lists_by_index(data_handler, context, messages, httpx_mock):
    list_one = json.loads(resource("list_of_idigbio_records.json"))
    httpx_mock.add_response(url="https://artifact.test/list_one", json=list_one)

    list_two = [{"new_field": "fee"}, {"new_field": "fi"}, {"new_field": "fo"}]
    httpx_mock.add_response(url="https://artifact.test/list_two", json=list_two)

    expected_list = [
        record_one | record_two for record_one, record_two in zip(list_one, list_two)
    ]

    await data_handler.run(
        context,
        "Combine these lists",
        "process_data",
        agent.EntrypointParameters(
            artifacts=[
                Artifact(
                    local_id="#abcd",
                    mimetype="application/json",
                    description="A list of occurrence records",
                    uris=["https://artifact.test/list_one"],
                    metadata={"source": "iDigBio", "length": 3},
                ),
                Artifact(
                    local_id="#1234",
                    mimetype="application/json",
                    description="A list of extra data",
                    uris=["https://artifact.test/list_two"],
                    metadata={"source": "iDigBio", "length": 3},
                ),
            ]
        ),
    )

    artifact_message = next((m for m in messages if isinstance(m, ArtifactResponse)))
    assert artifact_message

    new_list = json.loads(artifact_message.content.decode("utf-8"))

    assert new_list == expected_list
