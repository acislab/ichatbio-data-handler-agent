import json

from ichatbio.agent_response import IChatBioAgentProcess

from context import ValidatedArtifactID, current_context, current_artifacts
from tools.util import context_tool
from tools.util import retrieve_artifact_content


def _is_record(a):
    return type(a) is dict


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

        list_one = await retrieve_artifact_content(artifact_one, process)
        list_two = await retrieve_artifact_content(artifact_two, process)

        await process.log("Inferring the JSON data's schema")
        for artifact, content in (
            (artifact_one, list_one),
            (artifact_two, list_two),
        ):
            if not (type(content) is list) or not all(filter(_is_record, content)):
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
