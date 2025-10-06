import json
import logging
from typing import Union

import instructor
import jq
from ichatbio.agent_response import IChatBioAgentProcess, ResponseContext
from ichatbio.types import Artifact
from instructor import AsyncInstructor
from instructor.exceptions import InstructorRetryException
from langchain.tools import tool
from openai import AsyncOpenAI
from pydantic import Field, BaseModel, field_validator

from artifact_registry import ArtifactRegistry
from tools.util import JSON, retrieve_artifact_content
from tools.util import capture_messages
from tools.util import contains_non_null_content

MAX_CHARACTERS_TO_SHOW_AI = 1024 * 10
MAX_SOURCE_PREVIEW_SIZE = 500

NONE = object()


class JQQuery(BaseModel):
    plan: str = Field(
        description="A brief explanation of how you plan to query the data (what fields to use, any filters, transformations, etc.)"
    )
    jq_query_string: str = Field(
        description="A JQ query string to process the json artifact.",
    )
    output_description: str = Field(
        description="A concise characterization of the data that the query will retrieve",
        examples=[
            "List of collectors of Rattus rattus records in iDigBio",
            "GBIF occurrence records modified in 2025",
        ],
    )


class GiveUp(BaseModel):
    reason: str


def _make_validating_response_model(source_content: JSON, results_box: list):
    class ValidatedJQQuery(JQQuery):
        @field_validator("jq_query_string", mode="after")
        @classmethod
        def validate_jq_query_string(cls, query):
            # If the LLM doesn't know how to construct an appropriate query, it shouldn't generate one
            if not query:
                return query

            try:
                compiled = jq.compile(query)
            except ValueError as e:
                raise ValueError(f"Failed to compile JQ query string {query}", e)

            try:
                result = compiled.input_value(source_content).all()
            except ValueError as e:
                raise ValueError(
                    f"Failed to execute JQ query {query} on provided content", e
                )

            if not contains_non_null_content(result):
                raise ValueError(
                    "Executing the JQ query on the input data returned an empty result. Does the query string match the schema of the input data?"
                )

            results_box[0] = result

            return query

    class ResponseModel(BaseModel):
        response: Union[ValidatedJQQuery | GiveUp] = Field(
            description="The action you are going to take. If the request can be fulfilled by running a JQ query on data matching the given schema, then you should generate a JQ query. Otherwise, if the request does not make sense with the provided data (e.g. if there are no relevant fields), you should give up and explain why."
        )

    return ResponseModel


SYSTEM_PROMPT = """\
You generate JQ query strings to process json data. Only respond with a single query string with valid JQ syntax. The
user will also provide a description of the data.

# Guidelines

Do not add filters that are redundant with the user's description of the data being processed. For example:
- If the data are all records of the same species, do not filter them by that species

The provided data are likely to be non-normalized. For example, the names of species, countries, collections, 
institutions, etc. are unlikely to be written exactly the same way in different records. Unless you are certain that
field values are going to be consistent, prefer to use regex matching ("test" filters) instead of exact matching
("select" filters). If needed, you may consider testing for different variations of names that can be represented in
various ways. For example, a person's first name may be initialized or spelled out, country names may be acronyms or 
spelled out, etc. 
"""


async def _generate_and_run_jq_query(
    request: str, schema: dict, source_content: JSON, source_artifact: Artifact
) -> (JQQuery | GiveUp, JSON):
    source_meta = source_artifact.model_dump_json()
    preview = json.dumps(source_content)[:MAX_SOURCE_PREVIEW_SIZE]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f'The data have the following metadata: "{source_meta}"',
        },
        {
            "role": "user",
            "content": "The JSON data to be processed have the JSON Schema definition that follows. Assume that fields "
            'with no specified "type" are strings.\n\n'
            + json.dumps(schema).replace(': {"type": "string"}', ""),
        },  # Save some tokens
        {
            "role": "user",
            "content": f"For reference, here's the first {len(preview)} characters of the source data: {preview}",
        },
        {"role": "user", "content": request},
    ]

    results_box = [None]
    response_model = _make_validating_response_model(source_content, results_box)

    try:
        client: AsyncInstructor = instructor.from_openai(AsyncOpenAI())
        result = await client.chat.completions.create(
            model="gpt-4.1-unfiltered",
            temperature=0,
            response_model=response_model,
            messages=messages,
            max_retries=5,
        )
    except InstructorRetryException as e:
        logging.warning("Failed to generate JQ query string", e)
        raise

    response: JQQuery | GiveUp = result.response

    return response, results_box[0]


def is_record(a):
    return type(a) is dict


def make_tool(request: str, context: ResponseContext, artifacts: ArtifactRegistry):
    """
    The tool needs access to the `context` object in order to respond to iChatBio. To accomplish this, we define a new
    tool for each request, using the `context` object in its definition.
    """

    @tool("join_lists")
    async def run(artifact_one_id: str, artifact_two_id: str):
        """
        Joins two lists of equal length, joining items by index to produce a new list of the same length.

        :param artifact_one_id: A list artifact
        :param artifact_two_id: Another list artifact of the same length
        :return: A new artifact
        """
        artifact_one, artifact_two = artifacts.get(artifact_one_id, artifact_two_id)

        with capture_messages(context) as messages:
            await join_lists(artifact_one, context, artifact_two)
        return messages  # Pass the iChatBio messages back to the LangChain agent as context

    return run


async def join_lists(
    artifact_one: Artifact, context: ResponseContext, artifact_two: Artifact
):
    async with context.begin_process("Processing data") as process:
        process: IChatBioAgentProcess

        list_one = await retrieve_artifact_content(artifact_one, process)
        list_two = await retrieve_artifact_content(artifact_two, process)

        await process.log("Inferring the JSON data's schema")
        for artifact, content in (
            (artifact_one, list_one),
            (artifact_two, list_two),
        ):
            if not (type(content) is list) or not all(filter(is_record, content)):
                await process.log(
                    f"Error: artifact {artifact.local_id} is not a list of records"
                )
                return

        new_list = [
            record_one | record_two
            for record_one, record_two in zip(list_one, list_two)
        ]

        await process.create_artifact(
            mimetype="application/json",
            description=f"Joined list of records from artifacts {artifact_one.local_id} and {artifact_two.local_id}",
            content=json.dumps(new_list).encode("utf-8"),
            metadata={
                "source_artifacts": [artifact_one.local_id, artifact_two.local_id]
            },
        )
