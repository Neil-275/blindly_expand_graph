from src.utils import extract_strings, flatten_nested_tuple

from pydantic import BaseModel, computed_field
from typing import Union, Optional


class QueryEntry(BaseModel):
    query: tuple
    query_type: tuple
    answer_ids: list[str]

    @computed_field
    @property
    def start_entities(self) -> list[str]:
        ids = flatten_nested_tuple(self.query)
        notations = flatten_nested_tuple(self.query_type)
        return [
            id
            for index, id in enumerate(ids)
            if notations[index] == 'e'
        ]


class ReasoningPath(BaseModel):
    current_entity: str
    trace: list

    def __str__(self):
        # Redefine if needed for better visualization of the reasoning path
        return super().__str__()


class AnswerReachedCheckResult(BaseModel):
    reached: bool
    answer: Optional[list[str]] = None
    explanation: Optional[str] = None
