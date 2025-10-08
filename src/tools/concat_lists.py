import json

from ichatbio.agent_response import IChatBioAgentProcess, ResponseContext
from ichatbio.types import Artifact
from langchain.tools import tool

from artifact_registry import ArtifactRegistry
from tools.util import capture_messages
from tools.util import retrieve_artifact_content


def is_record(a):
    return type(a) is dict


def make_tool(request: str, context: ResponseContext, artifacts: ArtifactRegistry):
    """
    The tool needs access to the `context` object in order to respond to iChatBio. To accomplish this, we define a new
    tool for each request, using the `context` object in its definition.
    """

    @tool("concat_lists")
    async def run(
        artifact_one_id: artifacts.model,
        artifact_two_id: artifacts.model,
    ):
        """
        Concatenates two lists.

        :param artifact_one_id: A list artifact
        :param artifact_two_id: Another list artifact
        :return: A new artifact
        """
        artifact_one, artifact_two = artifacts.get(artifact_one_id, artifact_two_id)

        with capture_messages(context) as messages:
            await concat_lists(artifact_one, context, artifact_two)
        return messages  # Pass the iChatBio messages back to the LangChain agent as context

    return run


async def concat_lists(
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

        new_list = list_one + list_two

        await process.create_artifact(
            mimetype="application/json",
            description=f"Joined list of records from artifacts {artifact_one.local_id} and {artifact_two.local_id}",
            content=json.dumps(new_list).encode("utf-8"),
            metadata={
                "source_artifacts": [artifact_one.local_id, artifact_two.local_id]
            },
        )
