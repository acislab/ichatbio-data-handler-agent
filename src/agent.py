"""
This agent runs as a LangChain ReAct-style agent with enforced tool-calling. Such agents operate by iteratively calling
tools in order to fulfill the user's request. This is modeled as a conversation which begins with just the user's
request, then each subsequent toolcall appends agent-generated messages to the conversation. This particular agent has
access to two special tools - "abort" and "finish" - which the agent calls when it decides that either it has
successfully fulfilled the user's request ("finish") or that it isn't able to do so and should quit instead ("abort").

See the flowchart in README.md for a visualization of the agent.
"""

from typing import override, Iterable

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

from artifact_registry import ArtifactRegistry
from tools import process_data, join_lists

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


class EntrypointParameters(BaseModel):
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
                    id="process_data",
                    description=DESCRIPTION,
                    parameters=EntrypointParameters,
                )
            ],
        )

    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: EntrypointParameters,  # It's safe to assume type Parameter because we only have one entrypoint
    ):
        """
        Running this agent first builds a LangChain agent graph (a loop that alternates between decision-making and
        tool execution), then executes the graph with `request` as input. The tools themselves are responsible for
        sending messages back to iChatBio via the `context` object. To give the tools access to the request context, we
        instantiate new tools each time a request is received; this allows the agent to safely handle concurrent
        requests.
        """

        # Build the agent's tools

        @tool(return_direct=True)  # This tool ends the agent loop
        async def abort(reason: str):
            """If you can't fulfill the user's request, abort instead and explain why."""
            await context.reply(reason)

        @tool(return_direct=True)  # This tool ends the agent loop
        async def finish(message: str):
            """Mark the user's request as successfully completed."""
            await context.reply(message)

        artifacts = ArtifactRegistry(params.artifacts)
        tools = [
            process_data.make_tool(request, context, artifacts),
            join_lists.make_tool(request, context, artifacts),
            abort,
            finish,
        ]

        # Build a LangChain agent graph

        # TODO: make the LLM configurable
        llm = ChatOpenAI(model="gpt-4.1", tool_choice="required")

        system_message = make_system_message(params.artifacts)
        agent = langchain.agents.create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_message,
        )

        # Run the graph

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
"""


def list_artifact(artifact: Artifact):
    return f"""\
- local_id: {artifact.local_id}
  description: {artifact.description}
  uris: {artifact.uris}
  metadata: {artifact.metadata}\
"""


def make_system_message(artifacts: Iterable[Artifact]):
    return SYSTEM_MESSAGE.format(
        artifacts=(
            "\n\n".join([list_artifact(artifact) for artifact in artifacts])
            if artifacts
            else "NO AVAILABLE ARTIFACTS"
        )
    ).strip()


def create_app() -> Starlette:
    agent = DataHandlerAgent()
    app = build_agent_app(agent)
    return app
