import dotenv

import app_config

dotenv.load_dotenv()
app_config.reload()

app_config.llm_access.set(
    app_config.LLMAccess(
        base_url=app_config.settings().llm_base_url,
        api_key=app_config.settings().llm_api_key,
    )
)
