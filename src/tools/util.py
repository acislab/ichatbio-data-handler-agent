import functools
import traceback
import types
from contextlib import contextmanager

import httpx
import langchain.tools
from genson import SchemaBuilder
from genson.schema.strategies import Object
from ichatbio.agent_response import (
    ArtifactResponse,
    IChatBioAgentProcess,
    ResponseContext,
)
from ichatbio.types import Artifact

from context import current_context

JSON = dict | list | str | int | float | None
"""JSON-serializable primitive types that work with functions like json.dumps(). Note that dicts and lists may contain
content that is not JSON-serializable."""


def contains_non_null_content(content: JSON):
    """
    Returns True only if the JSON-serializable content contains a non-empty value. For example, returns True for [[1]]
    and False for [[]].
    """
    match content:
        case None:
            return False
        case list() as l:
            return any([contains_non_null_content(v) for v in l])
        case dict() as d:
            return any([contains_non_null_content(v) for k, v in d.items()])
        case _:
            return True


def context_tool(func):
    """
    Turns the function into a langchain tool that emits iChatBio messages.
    """

    @langchain.tools.tool(
        func.__name__,
        description=func.__doc__,
        return_direct=True,  # TODO: this ends the agent loop immediately. Remove this when the agent can use its own artifacts.
    )
    @functools.wraps(func)  # Preserves function signature
    async def wrapper(*args, **kwargs):
        context = current_context.get()
        with capture_messages(context) as messages:
            await func(*args, **kwargs)
            return messages  # Pass the iChatBio messages back to the LangChain agent as context

    return wrapper


@contextmanager
def capture_messages(context: ResponseContext):
    """
    Modifies a ResponseContext so that any messages sent back to iChatBio are also collected into a list.

    Usage:
        context: ResponseContext
        with capture_messages(context) as messages:
            await context.reply("Alert!")
            # Now messages[0] is a DirectResponse object
    """
    messages = []

    channel = context._channel
    old_submit = channel.submit

    async def submit_and_buffer(self, message):
        await old_submit(message)
        match message:
            case ArtifactResponse() as artifact:
                # Remove "content" from artifact messages, the AI doesn't need to see it:
                # - The artifact description and metadata provide enough context for decision-making
                # - If the AI sees content, it may process it directly instead of running reliable processes
                # - It can be expensive to include content in LLM context, and it might not even fit
                messages.append(
                    ArtifactResponse(
                        description=artifact.description,
                        mimetype=artifact.mimetype,
                        metadata=artifact.metadata,
                    )
                )
            case _:
                messages.append(message)

    channel.submit = types.MethodType(submit_and_buffer, channel)

    yield messages

    channel.submit = old_submit


# JSON schema extraction


class NoRequiredObject(Object):
    KEYWORDS = tuple(kw for kw in Object.KEYWORDS if kw != "required")

    # Remove "required" from the output if present
    def to_schema(self):
        schema = super().to_schema()
        if "required" in schema:
            del schema["required"]
        return schema


class NoRequiredSchemaBuilder(SchemaBuilder):
    """SchemaBuilder that does not use the "required" keyword, which roughly doubles the length of the schema string,
    and also isn't very helpful for our purposes."""

    EXTRA_STRATEGIES = (NoRequiredObject,)


def extract_json_schema(content: str) -> dict:
    builder = NoRequiredSchemaBuilder()
    builder.add_object(content)
    schema = builder.to_schema()
    return schema


def format_exception(e) -> str:
    return "; ".join(traceback.format_exception(e, limit=0))


async def retrieve_artifact_content(
    artifact: Artifact, process: IChatBioAgentProcess
) -> JSON:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as internet:
            for url in artifact.get_urls():
                # Rewrite URL if needed (for Docker networking)
                rewritten_url = _rewrite_localhost_url(url)
                if rewritten_url != url:
                    await process.log(
                        f"Rewriting URL for Docker networking: {url} -> {rewritten_url}"
                    )
                    url = rewritten_url

                await process.log(
                    f"Retrieving artifact {artifact.local_id} content from {url}"
                )
                response = await internet.get(url)
                if response.is_success:
                    return response.json()  # TODO: catch exception?
                else:
                    await process.log(
                        f"Error downloading artifact content: {response.reason_phrase} ({response.status_code})"
                    )
                    raise ValueError()
            else:
                await process.log(
                    "Failed to find where the artifact content is located"
                )
                raise ValueError()
    except httpx.HTTPError as e:
        await process.log(f"Error retrieving artifact content: {format_exception(e)}")


def _rewrite_localhost_url(url: str) -> str:
    """Rewrite localhost URLs to host.docker.internal for Docker compatibility."""
    return url.replace("localhost:", "host.docker.internal:").replace(
        "127.0.0.1:", "host.docker.internal:"
    )


async def retrieve_artifact_text(
    artifact: Artifact, process: IChatBioAgentProcess
) -> str | None:
    """Retrieves artifact content as raw text (for CSV and text formats)."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as internet:
            for url in artifact.get_urls():
                # Rewrite URL if needed (for Docker networking)
                rewritten_url = _rewrite_localhost_url(url)
                if rewritten_url != url:
                    await process.log(
                        f"Rewriting URL for Docker networking: {url} -> {rewritten_url}"
                    )
                    url = rewritten_url

                await process.log(
                    f"Retrieving artifact {artifact.local_id} content from {url}"
                )
                response = await internet.get(url)
                if response.is_success:
                    return response.text
                else:
                    await process.log(
                        f"Error downloading artifact content: {response.reason_phrase} ({response.status_code})"
                    )
            else:
                await process.log(
                    "Failed to find where the artifact content is located"
                )
    except httpx.HTTPError as e:
        await process.log(f"Error retrieving artifact content: {format_exception(e)}")

    return None
