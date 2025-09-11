import importlib.resources
import json

import jq
import pytest
import pytest_asyncio
from ichatbio.agent_response import DirectResponse, ProcessBeginResponse, ProcessLogResponse, ArtifactResponse, ResponseMessage
from ichatbio.types import Artifact

from src.agent import DataHandlerAgent
from src.entrypoints import process_json


@pytest.mark.skip(reason="Just for reference")
def test_jq():
    data = "1\n2\n3"
    result = jq.compile(".").input_value(data).first()
    pass


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
            metadata={}
        )
    )

    assert result == ["cricket"]


@pytest.mark.skip(reason="Just for reference")
def test_schema_generation():
    data = json.loads(importlib.resources.files("resources").joinpath("idigbio_records_search_result.json") \
                      .read_text())
    schema = process_json._generate_json_schema(data)
    pass


@pytest_asyncio.fixture()
def agent():
    return DataHandlerAgent()


@pytest.mark.httpx_mock(should_mock=lambda request: request.url == "https://artifact.test")
@pytest.mark.asyncio
async def test_extract_first_record(agent, context, messages, httpx_mock):
    source_data = json.loads(
        importlib.resources.files("resources").joinpath("idigbio_records_search_result.json").read_text()
    )
    httpx_mock.add_response(url="https://artifact.test", json=source_data)

    source_artifact = Artifact(
        local_id="#0000",
        mimetype="application/json",
        description="A list of occurrence records",
        uris=["https://artifact.test"],
        metadata={"source": "iDigBio"}
    )

    await agent.run(
        context,
        "Get the first record",
        "process_json",
        process_json.Parameters(artifacts=[source_artifact])
    )

    artifact_message = next((m for m in messages if isinstance(m, ArtifactResponse)))
    assert artifact_message

    first_record = json.loads(artifact_message.content.decode("utf-8"))

    # If the record was put into a single-item list, pull it out
    if type(first_record) is list:
        assert len(first_record) == 1
        first_record = first_record[0]

    assert first_record == source_data["items"][0]
