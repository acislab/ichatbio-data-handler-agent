import charset_normalizer
import filetype

from artifact_io.artifact_data import DataCodec, StructuredData
from artifact_io.csv_codec import CsvCodec
from artifact_io.json_codec import JsonCodec

_codec_lookup = {
    "application/json": JsonCodec,
    "application/geo+json": JsonCodec,
    "text/json": JsonCodec,
    "application/csv": CsvCodec,
    "application/x-csv": CsvCodec,
    "text/csv": CsvCodec,
    "text/tsv": CsvCodec,
    "text/comma-separated-values": CsvCodec,
    "text/tab-separated-values": CsvCodec,
    "text/x-comma-separated-values": CsvCodec,
    "text/x-tab-separated-values": CsvCodec,
    "text/x-csv": CsvCodec,
    "text/x-tsv": CsvCodec,
}


def get_codec(mimetype: str) -> DataCodec | None:
    if constructor := _codec_lookup.get(mimetype):
        return constructor()
    return None


def guess_type(binary: bytes) -> DataCodec | None:
    try:
        mimetype = filetype.guess_mime(binary)
        return get_codec(mimetype)
    except TypeError:
        return None


def get_encoding(binary: bytes) -> str:
    return charset_normalizer.detect(binary).get("encoding") or "utf-8"


def encode_artifact_content(data: StructuredData, codec: DataCodec) -> bytes:
    return codec.encode(data)
