import csv
import json
from io import StringIO
from typing import Any, Dict, List, Union

from ichatbio.agent_response import IChatBioAgentProcess

from context import (
    current_context,
    current_artifacts,
    ValidatedArtifactID,
)
from tools.util import retrieve_text_artifact

NONE = object()


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flattens a nested dictionary using dot notation."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(
                        _flatten_dict(item, f"{new_key}[{i}]", sep=sep).items()
                    )
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, v))
    return dict(items)


def _json_to_csv(data: Union[str, List[Dict], Dict]) -> str:
    """Converts JSON data to CSV format."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}")

    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("JSON must be an object or array of objects")

    if not records:
        return ""

    flattened_records = [_flatten_dict(record) for record in records]

    fieldnames = set()
    for record in flattened_records:
        fieldnames.update(record.keys())
    fieldnames = sorted(list(fieldnames))

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(flattened_records)

    return output.getvalue()


def _csv_to_json(csv_content: str) -> str:
    """Converts CSV data to JSON format."""
    if not csv_content.strip():
        return json.dumps([])

    reader = csv.DictReader(StringIO(csv_content))
    records = list(reader)

    if not records:
        return json.dumps([])

    return json.dumps(records, indent=2)


async def convert_json_csv(
    artifact_id: ValidatedArtifactID,
    output_format: str,
):
    """
    Convert between JSON and CSV formats.

    :param artifact_id: The source artifact containing data to convert
    :param output_format: The desired output format: 'csv' to convert JSON to CSV, or 'json' to convert CSV to JSON
    """
    context = current_context.get()
    artifacts = current_artifacts.get()

    source_artifact = artifacts.get(artifact_id)

    async with context.begin_process("Converting data format") as process:
        process: IChatBioAgentProcess

        artifact_content = await retrieve_text_artifact(source_artifact, process)
        if artifact_content is None:
            await process.log("Failed to retrieve data for processing")
            return

        try:
            await process.log(
                f"Converting artifact {source_artifact.local_id} to {output_format.upper()}"
            )

            if output_format.lower() == "csv":
                result = _json_to_csv(artifact_content)
                await process.log(f"Successfully converted JSON to CSV")

                await process.create_artifact(
                    mimetype="text/csv",
                    description="Converted CSV from JSON",
                    content=result.encode("utf-8"),
                    metadata={
                        "conversion_type": "json_to_csv",
                        "source_artifact": source_artifact.local_id,
                        "format": "csv",
                    },
                )

                await context.reply(
                    text=f"Successfully converted artifact {source_artifact.local_id} from JSON to CSV."
                )
            else:  # output_format == "json"
                result = _csv_to_json(artifact_content)
                await process.log(f"Successfully converted CSV to JSON")

                await process.create_artifact(
                    mimetype="application/json",
                    description="Converted JSON from CSV",
                    content=result.encode("utf-8"),
                    metadata={
                        "conversion_type": "csv_to_json",
                        "source_artifact": source_artifact.local_id,
                        "format": "json",
                    },
                )

                await context.reply(
                    text=f"Successfully converted artifact {source_artifact.local_id} from CSV to JSON."
                )
        except ValueError as e:
            await process.log(f"Conversion failed: {e}")
            await context.reply(f"Conversion failed: {e}")
        except Exception as e:
            await process.log(f"Unexpected error during conversion: {e}")
            await context.reply(f"Unexpected error during conversion: {e}")
