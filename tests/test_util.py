import importlib.resources
import json

import pytest
from artifact_io import csv_codec
from ichatbio.agent_response import ProcessBeginResponse, ArtifactResponse
from ichatbio.types import Artifact
from tests.conftest import resource
from tools import util
from tools.util import contains_non_null_content, extract_json_schema


class TestContainsNonNullContent:
    def test_no_content_is_null(self):
        assert not contains_non_null_content(None)

    def test_empty_list_is_null(self):
        assert not contains_non_null_content([])

    def test_empty_dict_is_null(self):
        assert not contains_non_null_content({})

    def test_empty_leaves_is_null(self):
        assert not contains_non_null_content([{}, []])

    def test_primitive_is_non_null(self):
        assert contains_non_null_content("haha")

    def test_empty_string_is_non_null(self):
        assert contains_non_null_content("")

    def test_partially_null_is_non_null(self):
        assert contains_non_null_content(["haha", None])

    def test_nested_is_non_null(self):
        assert contains_non_null_content({"key": {"key": "value"}})


def test_extract_json_schema():
    data = json.loads(
        importlib.resources.files("resources")
        .joinpath("idigbio_records_search_result.json")
        .read_text()
    )
    schema = extract_json_schema(data)
    top_level_fields = set(schema["properties"]["items"]["items"]["properties"].keys())
    assert top_level_fields == {"data", "etag", "indexTerms", "type", "uuid"}


class TestReadCSV:
    def test_no_header(self):
        """For now, we assume the first line is a header. That means it won't be read as a record."""
        data = list(csv_codec.read_csv("1,2"))
        assert data == []

    def test_ok(self):
        data = list(csv_codec.read_csv("a,b\n1,2\n3,4"))
        assert data == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]

    def test_tab_delimited(self):
        data = list(csv_codec.read_csv("a\tb\n1\t2\n3\t4"))
        assert data == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


class TestParseAny:
    def test_json(self):
        pass

    def test_csv(self):
        pass


OCCURRENCE_RECORDS = Artifact(
    local_id="#0000",
    mimetype="application/json",
    description="A list of occurrence records",
    uris=["https://artifact.test"],
    metadata={"source": "iDigBio"},
)


class TestRetrieval:
    @pytest.mark.httpx_mock(
        should_mock=lambda request: request.url == "https://artifact.test"
    )
    @pytest.mark.asyncio
    async def test_json(self, httpx_mock, context, messages):
        source_data = json.loads(resource("idigbio_records_search_result.json"))
        httpx_mock.add_response(url="https://artifact.test", json=source_data)

        async with context.begin_process("Processing") as process:
            await util.retrieve_artifact_data(OCCURRENCE_RECORDS, process)

        assert len(messages) > 1
        assert isinstance(messages[0], ProcessBeginResponse)
        assert not any([isinstance(message, ArtifactResponse) for message in messages])
