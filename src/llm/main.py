from src.config import settings
from src.models import ReasoningPath, AnswerReachedCheckResult
from src.llm.clients.base import LLMServiceBase
from src.llm.clients.openai import OpenAIInterface

from typing import TypeVar, Generic
from loguru import logger
import json


LLM_Type = TypeVar('LLM_Type', bound=LLMServiceBase)


def get_llm_instance(model_name: str) -> LLM_Type:
    model_prefix = model_name.split('-')[0]
    if model_prefix == 'gpt':
        return OpenAIInterface(model_name=model_name)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


class LLMInterface:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm_instance: LLM_Type = get_llm_instance(model_name)
        self.prompts = settings.prompts

    def check_path_reach_answers(
        self,
        question: str,
        formatted_triples: str
    ) -> AnswerReachedCheckResult:
        """
        This function checks if the current reasoning path can reach the answer(s).
        The core logic is to perform prompting to the LLM with question and current knowledge
        and ask the LLM to determine if the answer is reached.
        The output of LLM will be parsed to get the final result.

        Args:
            question (str): The natural language question being asked.
            path (ReasoningPath): The current reasoning path, which includes the trace of triples.
        Returns:
            AnswerReachedCheckResult: A structured result indicating whether the answer is reached
        """
        if not formatted_triples:
            logger.warning("Empty sub-knowledge graph. Cannot check if it reaches the answer.")
            return AnswerReachedCheckResult(
                reached=False,
                explanation="The sub knowledge graph is empty, so it cannot reach any answer."
            )
        else:
            system_prompt = self.prompts.check_answer_reached.system
            user_prompt = self.prompts.check_answer_reached.user.format(
                question=question,
                sub_knowledge_graph=formatted_triples.strip()
            )

            # This implements a simple retry when LLM did not response in the correct format
            while True:
                retry: int = 0
                try:
                    llm_response = self.llm_instance.run(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                    )
                    # Parse the LLM response to fill in the result
                    return AnswerReachedCheckResult(**json.loads(llm_response))
                except Exception as e:
                    retry += 1
                    if retry > 3:
                        logger.error(f"LLM response failed after 3 retries. Error: {e}")
                        raise e
                    logger.warning(
                        f"LLM response parsing failed. Retrying... Attempt {retry}."
                        f"Error: {e}"
                    )

    def find_suitable_path(
        self,
        question: str,
        entity_id: str,
        trace: list[tuple[str, str, str]],
        k: int = 3
    ):
        pass

    def run(self):
        self.llm_instance.run()
        pass


if __name__ == "__main__":
    llm_interface = LLMInterface(model_name='gpt-3.5-turbo')
    print(type(llm_interface.llm_instance))
