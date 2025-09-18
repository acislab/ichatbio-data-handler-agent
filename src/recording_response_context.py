from typing import override

from ichatbio.agent_response import ResponseChannel, ResponseContext, ResponseMessage


class RecordingResponseContext(ResponseContext):
    message_buffer: list[ResponseMessage] = []

    def __init__(self, context: ResponseContext, message_buffer: list[ResponseMessage]):
        class RecordingResponseChannel(ResponseChannel):
            @override
            def submit(self, message: ResponseMessage, context_id: str):
                message_buffer.append(message)

        super().__init__(RecordingResponseChannel())
        self._context = context
