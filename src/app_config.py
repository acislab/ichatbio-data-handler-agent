from contextvars import ContextVar

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    llm_base_url: str | None = Field(None, alias="OPENAI_BASE_URL")
    llm: str = Field("gpt-5.6-terra")
    reasoning_effort: str | None = Field("high")
    use_llm_proxy: bool = Field(True)
    agent_base_url: str | None = Field(None)


def settings() -> Settings:
    return Settings.settings_cached()


def reload():
    Settings.settings_clear_cache()


class LLMAccess(BaseModel):
    base_url: str | None = settings().llm_base_url
    api_key: str | None = settings().llm_api_key


llm_access: ContextVar[LLMAccess] = ContextVar("request_llm_credentials")


def set_llm_credentials(metadata: dict | None):
    if metadata and "https://ichatbio.org/a2a/v1" in metadata and settings().use_llm_proxy:
        proxy_metadata = metadata.get("https://ichatbio.org/a2a/v1", {})
        base_url = proxy_metadata.get("ichatbio_base_url", "")
        api_key = proxy_metadata.get("temporary_llm_key", "")

        if not (base_url and api_key):
            raise ValueError(
                f"Failed to apply LLM proxy configuration; missing required fields. Received configuration: {proxy_metadata}"
            )

        # TODO: the backend should directly send the proxy URL
        base_url = base_url.rstrip("/") + "/llm"
    else:
        base_url = settings().llm_base_url
        api_key = settings().llm_api_key
        if not api_key:
            raise ValueError(f"Missing LLM API key environment variable. Please set OPENAI_API_KEY or LLM_API_KEY!")

    llm_access.set(LLMAccess(base_url=base_url, api_key=api_key))
