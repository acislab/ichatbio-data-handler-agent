import json

import dotenv
import ichatbio.agent_response
import pytest
from ichatbio.agent_response import (
    ArtifactResponse,
)
from ichatbio.types import Artifact

from conftest import resource
from tools import process_data

dotenv.load_dotenv()

OCCURRENCE_RECORDS = Artifact(
    local_id="#0000",
    mimetype="application/json",
    description="A list of occurrence records",
    uris=["https://artifact.test"],
    metadata={"source": "iDigBio"},
)


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url == "https://artifact.test"
)
@pytest.mark.asyncio
async def test_extract_first_record(context, messages, httpx_mock):
    source_data = json.loads(resource("idigbio_records_search_result.json"))
    httpx_mock.add_response(url="https://artifact.test", json=source_data)

    # TODO: mock OpenAI

    await process_data.process_data(context, "Get the first record", OCCURRENCE_RECORDS)

    artifact_message = next((m for m in messages if isinstance(m, ArtifactResponse)))
    assert artifact_message

    first_record = json.loads(artifact_message.content.decode("utf-8"))

    # If the record was put into a single-item list, pull it out
    if type(first_record) is list:
        assert len(first_record) == 1
        first_record = first_record[0]

    assert first_record == source_data["items"][0]


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url == "https://artifact.test"
)
@pytest.mark.asyncio
async def test_abort_after_failed_query(context, messages, httpx_mock):
    source_data = json.loads(resource("idigbio_records_search_result.json"))
    httpx_mock.add_response(url="https://artifact.test", json=source_data)

    # TODO: mock OpenAI

    await process_data.process_data(
        context,
        'Extract the "dwc.moonphases" field from these records',
        OCCURRENCE_RECORDS,
    )

    assert len(messages) > 1
    assert isinstance(messages[0], ichatbio.agent_response.ProcessBeginResponse)
    assert not any([isinstance(message, ArtifactResponse) for message in messages])


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url == "https://artifact.test"
)
@pytest.mark.asyncio
async def test_extract_a_list(context, messages, httpx_mock):
    source_data = json.loads(resource("idigbio_records_search_result.json"))
    httpx_mock.add_response(url="https://artifact.test", json=source_data)

    # TODO: mock OpenAI

    await process_data.process_data(context, "Extract records list", OCCURRENCE_RECORDS)

    artifact_message = next((m for m in messages if isinstance(m, ArtifactResponse)))
    assert artifact_message

    records = json.loads(artifact_message.content.decode("utf-8"))

    assert records == source_data["items"]
