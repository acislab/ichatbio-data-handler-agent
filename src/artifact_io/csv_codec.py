import csv
from io import StringIO
from typing import Iterator

import artifact_io.util
from artifact_io.artifact_data import DataCodec, StructuredData


class CsvCodec(DataCodec):
    def __init__(self, text_encoding="utf-8"):
        self.text_encoding = text_encoding

    def decode(self, binary: bytes) -> StructuredData:
        encoding = artifact_io.util.get_encoding(binary)
        content = read_csv(binary.decode(encoding))
        return list(content)

    def encode(self, data: StructuredData) -> bytes:
        buffer = StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerows(data)
        return buffer.getvalue().encode(self.text_encoding)


def read_csv(text: str) -> Iterator[dict]:
    """Assumes the first line is the header"""
    dialect = csv.Sniffer().sniff(text)
    reader = csv.DictReader(StringIO(text), dialect=dialect)
    return reader
