from typing import override, Optional

import dotenv
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard
from pydantic import BaseModel
from starlette.applications import Starlette

from .entrypoints import process_json

dotenv.load_dotenv()

DESCRIPTION = """\
Reads and processes JSON artifacts. This agent can do the following:
- Extract a set of fields from an object
- Change field names
- Display items in a list
- Filter items in a list
- etc.

To use this agent, provide an artifact local_id and describe how the json data should be filtered or transformed.

This tool works for both JSON objects and JSON lists. Under the hood, this agent uses JQ (https://jqlang.org/), a
lightweight JSON processing tool.
"""


class DataHandlerAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="JSON Handler",
            description=DESCRIPTION,
            icon=None,
            entrypoints=[process_json.entrypoint]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        match entrypoint:
            case process_json.entrypoint.id:
                await process_json.run(context, request, params)
            case _:
                raise ValueError()


def create_app() -> Starlette:
    agent = DataHandlerAgent()
    app = build_agent_app(agent)
    return app
