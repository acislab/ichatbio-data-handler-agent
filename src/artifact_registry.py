from typing import Annotated

from ichatbio.types import Artifact
from pydantic import Field

ArtifactID = Annotated[str, Field(pattern="^#[0-9a-f]{4}$", examples=["#01ef"])]


class ArtifactRegistry:
    def __init__(self, artifacts: list[Artifact]):
        self._artifacts = {artifact.local_id: artifact for artifact in artifacts}
        self.model = ArtifactID

    def get(self, aid: ArtifactID) -> Artifact:
        """
        Retrieves the Artifact identified by each id.

        Raises:
             ValueError: if artifact content can't be retrieved
        """
        artifact = self._artifacts.get(aid)
        if artifact is None:
            raise ValueError(f"Registry does not contain artifact {aid}")
        return artifact

    def get_all(self, *ids: ArtifactID) -> tuple[Artifact, ...]:
        """
        Retrieves the Artifact identified by each id. Raises a ValueError if an artifact doesn't exist.
        """
        return tuple((self._artifacts.get(i) for i in ids))
