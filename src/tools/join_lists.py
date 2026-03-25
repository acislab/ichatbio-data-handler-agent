import json

from ichatbio.agent_response import IChatBioAgentProcess

from context import ValidatedArtifactID, current_context, current_artifacts
from tools.util import context_tool, ProcessError, retrieve_json_list_artifact


@context_tool
async def join_lists(
    artifact_one_id: ValidatedArtifactID, artifact_two_id: ValidatedArtifactID
):
    """
    Joins two lists of equal length, joining items by index to produce a new list of the same length.

    :param artifact_one_id: A list artifact
    :param artifact_two_id: Another list artifact of the same length
    :return: A new artifact
    """

    context = current_context.get()
    artifacts = current_artifacts.get()

    artifact_one, artifact_two = artifacts.get(artifact_one_id, artifact_two_id)

    async with context.begin_process("Processing data") as process:
        process: IChatBioAgentProcess

        try:
            list_one = await retrieve_json_list_artifact(artifact_one, process)
            list_two = await retrieve_json_list_artifact(artifact_two, process)
        except ProcessError:
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
