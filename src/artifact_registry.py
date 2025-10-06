from ichatbio.types import Artifact


class ArtifactRegistry:
    def __init__(self, artifacts: list[Artifact]):
        self._artifacts = {artifact.local_id: artifact for artifact in artifacts}

    def get(self, *ids) -> Artifact | tuple[Artifact]:
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
