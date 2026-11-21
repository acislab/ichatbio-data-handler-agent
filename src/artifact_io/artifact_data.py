from abc import ABC, abstractmethod

StructuredData = dict | list | str | int | float | None
"""JSON-serializable primitive types that work with functions like json.dumps(). Note that dicts and lists might contain
content that is not JSON-serializable."""


class DataCodec(ABC):
    @abstractmethod
    def decode(self, binary: bytes) -> StructuredData:
        ...

    @abstractmethod
    def encode(self, data: StructuredData) -> bytes:
        ...


class CodecError(Exception):
    pass
