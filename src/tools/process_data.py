import json
import logging
from typing import Union

import instructor
import jq
from genson import SchemaBuilder
from genson.schema.strategies import Object
from httpx import AsyncClient
from ichatbio.agent_response import IChatBioAgentProcess, ResponseContext
from ichatbio.types import Artifact
from instructor import AsyncInstructor
from instructor.exceptions import InstructorRetryException
from langchain.tools import tool
from openai import AsyncOpenAI
from pydantic import Field, BaseModel, field_validator

from tools.tool import capture_messages
from util import contains_non_null_content

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


def make_tool(request: str, context: ResponseContext, artifacts: dict[str, Artifact]):
    @tool("process_data")
    async def run(artifact_id: str):
        """
        Filter or transform artifact data.

        :param artifact_id:
        :return: The result as a new artifact
        """
        with capture_messages(context) as messages:
            async with context.begin_process("Processing data") as process:
                process: IChatBioAgentProcess

                await process.log("Retrieving artifact data")
                source_artifact = artifacts.get(artifact_id)
                source_content = NONE

                async with AsyncClient(follow_redirects=True) as http:
                    for url in source_artifact.get_urls():
                        await process.log(
                            f"Retrieving artifact {source_artifact.local_id} content from {url}"
                        )
                        response = await http.get(url)
                        if response.is_success:
                            source_content = response.json()  # TODO: catch exception?
                            break

                if source_content is NONE:
                    await process.log("Failed to retrieve data for processing")
                    return

                await process.log("Inferring the JSON data's schema")
                schema = _generate_json_schema(source_content)

                await process.log("Generating JQ query string")
                try:
                    # jq_query, output_description, query_result, plan
                    ai_action, generated_content = await _generate_and_run_jq_query(
                        request, schema, source_content, source_artifact
                    )
                except InstructorRetryException as e:
                    await process.log("Failed to generate JQ query string")
                    return

                match ai_action:
                    case GiveUp(reason=reason):
                        await process.log(
                            f"Refused to generate a JQ query string: " + reason
                        )
                    case JQQuery(
                        plan=plan,
                        jq_query_string=jq_query_string,
                        output_description=artifact_description,
                    ):
                        await process.log(f"*Plan: {plan}*")

                        await process.log(
                            "Generated JQ query", data={"query_string": jq_query_string}
                        )
                        output_as_bytes = json.dumps(generated_content).encode("utf-8")
                        output_size_in_bytes = len(output_as_bytes)

                        await process.log(
                            f"Executed JQ query generated {output_size_in_bytes} bytes of data"
                        )

                        new_artifact = dict(
                            mimetype="application/json",
                            description=artifact_description,
                            content=output_as_bytes,
                            metadata={
                                "source_artifact": source_artifact.local_id,
                                "source_jq_query": jq_query_string,
                            },
                        )

                        await process.create_artifact(**new_artifact)

            return messages

    return run


class NoRequiredObject(Object):
    KEYWORDS = tuple(kw for kw in Object.KEYWORDS if kw != "required")

    # Remove "required" from the output if present
    def to_schema(self):
        schema = super().to_schema()
        if "required" in schema:
            del schema["required"]
        return schema


class NoRequiredSchemaBuilder(SchemaBuilder):
    """SchemaBuilder that does not use the "required" keyword."""

    EXTRA_STRATEGIES = (NoRequiredObject,)


def _generate_json_schema(content: str) -> dict:
    builder = NoRequiredSchemaBuilder()
    builder.add_object(content)
    schema = builder.to_schema()
    return schema


def make_response_model(source_content: dict | list, results_box: list):
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


async def _generate_and_run_jq_query(
    request: str, schema: dict, source_content: dict | list, source_artifact: Artifact
) -> (str, str, dict):
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
    response_model = make_response_model(source_content, results_box)

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
