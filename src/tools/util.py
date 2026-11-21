import functools
import traceback
import types
from contextlib import contextmanager

import artifact_io.util
import httpx
import langchain.tools
from artifact_io.artifact_data import DataCodec, CodecError, StructuredData, ArtifactData
from context import current_context
from genson import SchemaBuilder
from genson.schema.strategies import Object
from ichatbio.agent_response import (
    ArtifactResponse,
    IChatBioAgentProcess,
    ResponseContext,
)
from ichatbio.types import Artifact


def contains_non_null_content(content: StructuredData):
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
        return_direct=True,
        # TODO: this ends the agent loop immediately. Remove this when the agent can use its own artifacts.
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


def extract_json_schema(content: StructuredData) -> dict:
    builder = NoRequiredSchemaBuilder()
    builder.add_object(content)
    schema = builder.to_schema()
    return schema


def format_exception(e) -> str:
    return "; ".join(traceback.format_exception(e, limit=0))


class ProcessError(Exception):
    """Something went wrong during an agent process."""


async def retrieve_artifact_bytes(
        artifact: Artifact, process: IChatBioAgentProcess
) -> bytes:
    """Retrieves artifact content as raw text (for CSV and text formats)."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as internet:
            for url in artifact.get_urls():
                await process.log(
                    f"Retrieving artifact {artifact.local_id} content from {url}"
                )
                response = await internet.get(url)
                if response.is_success:
                    await process.log(f"{len(response.content)} bytes received")
                    return response.content
                else:
                    await process.log(
                        f"Error downloading artifact content: {response.reason_phrase} ({response.status_code})"
                    )
                    raise ProcessError()

            else:
                await process.log("Failed to find resolvable URL among artifact's URIs")
                raise ProcessError()
    except httpx.HTTPError as e:
        await process.log(f"Error retrieving artifact content: {format_exception(e)}")
        raise ProcessError() from e


async def _decode(binary: bytes, try_codec: DataCodec | None, process: IChatBioAgentProcess) -> ArtifactData:
    codec = try_codec if try_codec else artifact_io.util.guess_type(binary)
    try:
        return codec.decode(binary)
    except CodecError:
        await process.log(f"Failed to read artifact content as {codec.mimetype}")
        if try_codec:
            return await _decode(binary, None, process)
        else:
            raise ProcessError()


async def retrieve_artifact_data(artifact: Artifact, process: IChatBioAgentProcess) -> ArtifactData | None:
    binary = await retrieve_artifact_bytes(artifact, process)
    if not binary:
        await process.log("Artifact has no content")
        return None

    codec = artifact_io.util.get_codec(artifact.mimetype)
    return await _decode(binary, codec, process)


async def retrieve_json_list_artifact(
        artifact: Artifact, process: IChatBioAgentProcess
) -> list:
    data = await retrieve_artifact_data(artifact, process)
    if not (data and isinstance(data.content, list)):
        await process.log(f"Artifact {artifact.local_id} content is not a list")
        raise ProcessError()
    return data.content
