from typing import override

import dotenv
import langchain.agents
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint, Artifact
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic import Field
from starlette.applications import Starlette

from tools import process_data

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


class Parameters(BaseModel):
    artifacts: list[Artifact] = Field(
        description="The JSON data to process", min_length=1
    )


class DataHandlerAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Data Handler",
            description=DESCRIPTION,
            icon=None,
            entrypoints=[
                AgentEntrypoint(
                    id="process_data", description=DESCRIPTION, parameters=Parameters
                )
            ],
        )

    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: Parameters,  # It's safe to assume type Parameter because we only have one entrypoint
    ):
        artifacts = {artifact.local_id: artifact for artifact in params.artifacts}
        tools = [process_data.make_tool(request, context, artifacts), abort]

        llm = ChatOpenAI(model="gpt-4.1-mini", tool_choice="required")

        system_message = make_system_message(artifacts)
        agent = langchain.agents.create_agent(
            model=llm,
            tools=tools,
            prompt=system_message,
        )

        await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": request},
                ]
            }
        )


SYSTEM_MESSAGE = """
You manipulate structured data using tools. You can access the following artifacts:

{artifacts}

If you are unable to fulfill the user's request using your available tools, abort and explain why.
""".strip()


def make_system_message(artifacts: dict[str, Artifact]):
    return SYSTEM_MESSAGE.format(
        artifacts=(
            "\n\n".join(
                [
                    f"- {id}: {artifact.model_dump_json()}"
                    for id, artifact in artifacts.items()
                ]
            )
            if artifacts
            else "NO AVAILABLE ARTIFACTS"
        )
    )


@tool(return_direct=True)  # This tool ends the agent loop
async def abort(reason: str):
    """If you can't do what was asked, abort instead and explain why."""
    pass


def create_app() -> Starlette:
    agent = DataHandlerAgent()
    app = build_agent_app(agent)
    return app
