import io
import json

from artifact_io.artifact_data import DataCodec, StructuredData


class JsonCodec(DataCodec):
    def __init__(self, text_encoding):
        self.text_encoding = text_encoding

    def decode(self, binary: bytes) -> StructuredData:
        content = json.load(io.BytesIO(binary))
        return content

    def encode(self, data: StructuredData) -> bytes:
        return json.dumps(data).encode(self.text_encoding)
