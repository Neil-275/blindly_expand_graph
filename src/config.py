from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from yaml import safe_load


# Core modules configurations
class GNNConfig(BaseModel):
    n_layer: int = 2
    hidden_dim: int = 128


class ProjecttorConfig(BaseModel):
    out_dim: int = 128


# Prompt configurations
class PromptEntry(BaseModel):
    system: str
    user: str | None = None


class PromptConfig(BaseModel):
    check_answer_reached: PromptEntry
    find_path: PromptEntry


# OpenAI API configurations
class LLMSecretConfig(BaseModel):
    openai_api_key: str
    llama_api_key: str


# The whole settings
class Settings(BaseSettings):
    prompts: PromptConfig = PromptConfig(**safe_load(open("src/llm/prompts.yaml", "r")))
    llm_secrets: LLMSecretConfig = LLMSecretConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )
    pass


# Singleton settings instance
settings = Settings()
