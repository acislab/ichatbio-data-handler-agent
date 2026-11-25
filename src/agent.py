"""
This agent runs as a LangChain ReAct-style agent with enforced tool-calling. Such agents operate by iteratively calling
tools in order to fulfill the user's request. This is modeled as a conversation which begins with just the user's
request, then each subsequent toolcall appends agent-generated messages to the conversation. This particular agent has
access to two special tools - "abort" and "finish" - which the agent calls when it decides that either it has
successfully fulfilled the user's request ("finish") or that it isn't able to do so and should quit instead ("abort").

See the flowchart in README.md for a visualization of the agent.
"""

from typing import Any, Iterable, override

import dotenv
import langchain.agents
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint, Artifact
from langchain.agents.middleware import after_model
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, Field
from starlette.applications import Starlette

import resources
from app_config import set_llm_credentials, llm_access, settings, reload
from artifact_registry import ArtifactRegistry
from context import current_artifacts, current_context, current_request
from tools.concat_lists import concat_lists
from tools.convert_json_csv import convert_json_csv
from tools.join_lists import join_lists
from tools.process_data import process_data

DESCRIPTION = """\
Reads and processes JSON artifacts. This agent can do the following:
- Extract a set of fields from an object
- Change field names
- Display items in a list
- Filter items in a list
- Join records across lists of the same length
- Concatenate lists
- Convert between JSON and CSV formats
- etc.

To use this agent, provide an artifact local_id and describe how the json data should be filtered, transformed, or converted.

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
            icon="https://raw.githubusercontent.com/acislab/ichatbio/refs/heads/main/icon.svg",
            documentation_url="https://github.com/acislab/ichatbio-data-handler-agent",
            url=settings().agent_base_url,
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
            params: BaseModel | None,
            metadata: dict[str, Any] | None = None,
    ):
        """
        Running this agent first builds a LangChain agent graph (a loop that alternates between decision-making and
        tool execution), then executes the graph with `request` as input. The tools themselves are responsible for
        sending messages back to iChatBio via the `context` object. To give the tools access to the request context, we
        instantiate new tools each time a request is received; this allows the agent to safely handle concurrent
        requests.
        """
        params: EntrypointParameters  # It's safe to assume type Parameter because we only have one entrypoint

        set_llm_credentials(metadata)
        artifacts = ArtifactRegistry(params.artifacts)
        current_request.set(request)
        current_context.set(context)
        current_artifacts.set(artifacts)

        control_loop_prompt = (resources.dir / "control_loop_prompt.md").read_text()
        system_message = make_system_message(control_loop_prompt, params.artifacts)

        # Run the graph
        langchain_agent = self._build_langchain_agent()
        await langchain_agent.ainvoke(
            {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": request},
                ]
            }
        )

    def _build_langchain_agent(self):
        @after_model
        async def share_text_messages(state, runtime):
            message = state["messages"][-1]
            if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content:
                await current_context.get().reply(message.content)

        return langchain.agents.create_agent(
            model=ChatOpenAI(
                model=settings().llm,
                streaming=True,
                base_url=llm_access.get().base_url,
                api_key=llm_access.get().api_key,
                reasoning_effort=settings().reasoning_effort,
                temperature=0,
                model_kwargs={"tool_choice": "required"}
            ),
            tools=[
                process_data,
                join_lists,
                concat_lists,
                convert_json_csv,
                abort,
                finish,
            ],
            middleware=[share_text_messages]
        )


@tool(return_direct=True)  # This tool ends the agent loop
async def abort(reason: str, runtime: ToolRuntime):
    """If you can't fulfill the user's request, abort instead and explain why."""
    await current_context.get().reply(reason)


@tool(return_direct=True)  # This tool ends the agent loop
async def finish(message: str, runtime: ToolRuntime):
    """Mark the user's request as successfully completed."""
    await current_context.get().reply(message)


def list_artifact(artifact: Artifact):
    return f"""\
- local_id: {artifact.local_id}
  description: {artifact.description}
  uris: {artifact.uris}
  metadata: {artifact.metadata}\
"""


def make_system_message(template: str, artifacts: Iterable[Artifact]):
    return template.format(
        artifacts=(
            "\n\n".join([list_artifact(artifact) for artifact in artifacts])
            if artifacts
            else "NO AVAILABLE ARTIFACTS"
        )
    ).strip()


def create_app() -> Starlette:
    dotenv.load_dotenv()
    reload()
    agent = DataHandlerAgent()
    app = build_agent_app(agent)
    return app
