from src.llm.clients.base import LLMServiceBase

from openai import OpenAI


class OpenAIInterface(LLMServiceBase):
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key="place-your-api-key-here")
        super().__init__(model_name=model_name)

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 2048
    ):
        messages = self._prepare_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        llm_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        choice = llm_response.choices[0]
        if choice.finish_reason != "stop":
            raise ValueError(f"LLM response not finished properly: {llm_response}")

        return choice.message.content
