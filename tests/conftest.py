import importlib
from importlib.resources.abc import Traversable

import app_config
import dotenv

dotenv.load_dotenv()
app_config.reload()

app_config.llm_access.set(
    app_config.LLMAccess(base_url=app_config.settings().llm_base_url, api_key=app_config.settings().llm_api_key)
)

def resource(*path, text=True) -> str | Traversable:
    file = importlib.resources.files("resources").joinpath(*path)
    if text:
        return file.read_text()
    return file
