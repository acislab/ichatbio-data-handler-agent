import importlib.resources
import json

import pytest
import pytest_asyncio
from ichatbio.agent_response import (
    ArtifactResponse,
)
from ichatbio.types import Artifact

import agent
from src.agent import DataHandlerAgent


@pytest.mark.asyncio
async def test__generate_jq_query():
    jq_query, description, result = await process_json._generate_jq_query(
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


@pytest.mark.skip(reason="Just for reference")
def test_schema_generation():
    data = json.loads(
        importlib.resources.files("resources")
        .joinpath("idigbio_records_search_result.json")
        .read_text()
    )
    schema = process_json._generate_json_schema(data)
    pass


@pytest_asyncio.fixture()
def the_agent():
    return DataHandlerAgent()


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url == "https://artifact.test"
)
@pytest.mark.asyncio
async def test_extract_first_record(the_agent, context, messages, httpx_mock):
    source_data = json.loads(
        importlib.resources.files("resources")
        .joinpath("idigbio_records_search_result.json")
        .read_text()
    )
    httpx_mock.add_response(url="https://artifact.test", json=source_data)

    source_artifact = Artifact(
        local_id="#0000",
        mimetype="application/json",
        description="A list of occurrence records",
        uris=["https://artifact.test"],
        metadata={"source": "iDigBio"},
    )

    await the_agent.run(
        context,
        "Get the first record",
        "process_json",
        agent.Parameters(artifacts=[source_artifact]),
    )

    artifact_message = next((m for m in messages if isinstance(m, ArtifactResponse)))
    assert artifact_message

    first_record = json.loads(artifact_message.content.decode("utf-8"))

    # If the record was put into a single-item list, pull it out
    if type(first_record) is list:
        assert len(first_record) == 1
        first_record = first_record[0]

    assert first_record == source_data["items"][0]


def test_system_message():
    artifact = Artifact(
        local_id="#0000",
        mimetype="application/json",
        description="A list of occurrence records",
        uris=["https://artifact.test"],
        metadata={"source": "iDigBio"},
    )

    system_message = agent.make_system_message({"#0000": artifact, "#1111": artifact})

    expected = """\
You manipulate structured data using tools. You can access the following artifacts:

- #0000: {"local_id":"#0000","description":"A list of occurrence records","mimetype":"application/json","uris":["https://artifact.test"],"metadata":{"source":"iDigBio"}}

- #1111: {"local_id":"#0000","description":"A list of occurrence records","mimetype":"application/json","uris":["https://artifact.test"],"metadata":{"source":"iDigBio"}}

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
