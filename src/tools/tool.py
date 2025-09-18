import types
from contextlib import contextmanager

from ichatbio.agent_response import ResponseContext


class Tool:
    def run(self, *args):
        pass


@contextmanager
def capture_messages(context: ResponseContext):
    messages = []

    channel = context._channel
    old_submit = channel.submit

    async def submit_and_buffer(self, message, context_id: str):
        await old_submit(message, context_id)
        messages.append(message)

    channel.submit = types.MethodType(submit_and_buffer, channel)

    yield messages

    channel.submit = old_submit
