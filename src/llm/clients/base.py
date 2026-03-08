from abc import ABC, abstractmethod
from typing import Optional


class LLMServiceBase(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _prepare_messages(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> list[dict[str, str]]:
        if not system_prompt and not user_prompt:
            raise ValueError("At least one of system_prompt or user_prompt must be provided.")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        return messages

    @abstractmethod
    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 2048
    ):
        pass
