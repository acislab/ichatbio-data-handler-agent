from typing import Annotated

import pydantic
from ichatbio.types import Artifact

ArtifactID = Annotated[
    str, pydantic.Field(pattern="^#[0-9a-f]{4}$", examples=["#01ef"])
]


class ArtifactRegistry:
    def __init__(self, artifacts: list[Artifact]):
        self._artifacts = {artifact.local_id: artifact for artifact in artifacts}
        self.model = make_validating_artifact_id_model(self)

    def get(self, *ids: ArtifactID) -> Artifact | tuple[Artifact]:
        """
        Retrieves the Artifact identified by each id. Raises a ValueError if an artifact doesn't exist.
        """
        id_artifacts = {i: self._artifacts.get(i) for i in ids}

        missing_artifacts = set(
            (i for i, artifact in id_artifacts.items() if artifact is None)
        )
        if missing_artifacts:
            raise ValueError(f"Unrecognized artifact ID(s): {missing_artifacts}")

        artifacts = tuple(id_artifacts.values())
        return artifacts if len(ids) > 1 else artifacts[0]


def make_validating_artifact_id_model(artifacts: ArtifactRegistry):
    def check_artifact(local_id: str):
        artifacts.get(
            ArtifactID(local_id)
        )  # Raises an exception if artifact doesn't exist
        return local_id

    ValidatedArtifactID = Annotated[
        str,
        pydantic.Field(pattern="^#[0-9a-f]{4}$", examples=["#01ef"]),
        pydantic.AfterValidator(check_artifact),
    ]

    return ValidatedArtifactID
