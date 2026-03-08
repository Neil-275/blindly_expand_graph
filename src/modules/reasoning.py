from src.kb_interface.freebase import freebase_interface
from src.modules.expand_subgraph import SubgraphSampler
from src.models import QueryEntry, ReasoningPath
from src.llm.main import LLMInterface

from typing import Optional
from loguru import logger


class ReasoningModule:
    def __init__(
        self,
        subgraph_sampler: SubgraphSampler,
        llm_model: str = 'gpt-3.5-turbo',
        gnn_model=None,
        projector_model=None,
        k: int = 4,
        max_hops: int = 4
    ):
        # Initialize with subgraph sampler and LLM interface
        self.subgraph_sampler = subgraph_sampler
        self.llm_interface = LLMInterface(model_name=llm_model)
        self.gnn_model = gnn_model
        self.projector_model = projector_model

        # Module-specific parameters
        self.k = k
        self.max_hops = max_hops

        # Cache for knowledge retrieval to avoid redundant calls
        self._mid2name_cache = {}
        self._id2rel_cache = {}

        # Check parameters
        if not self.llm_interface:
            logger.warning("ReasoningMoodule initialized without LLMInterface.")
        pass

    def _id_to_name(self, id: str) -> str:
        """
        Convert an entity mid to its human-readable name using the Freebase interface.
        """
        if id in self._mid2name_cache:
            return self._mid2name_cache[id]

        name = freebase_interface.convert_id_to_name(id)
        self._mid2name_cache[id] = name
        return name

    def _trace_to_string(self, trace: list[tuple[str, str, str]]):
        return "\n".join(
            f"{self._id_to_name(head)} "
            f"{relation} "
            f"{self._id_to_name(tail)}"

            for head, relation, tail in trace
        )

    def reset(self):
        """
        This function resets the reasoning module to its initial state,
        clearing any assigned query and resetting the subgraph sampler.
        Make it ready for a new query.
        """
        pass

    def assign_query(self, query: QueryEntry):
        self.query = query
        self.subgraph_sampler.assign_query(query.model_dump())
        pass

    def reasoning(self):
        """
        This function performs the reasoning process, which includes:
        Step-by-Step:
            1.
        """

        answers: list[str] = []

        active_paths = [
            ReasoningPath(
                current_entity=entity,
                trace=[entity]
            )
            for entity in self.query.start_entities
        ]

        # Loop over hops
        for hop in range(self.max_hops):
            logger.info(f"Loop into hop {hop + 1} with {len(active_paths)} active path(s).")
            new_path = []

            # Loop over active paths
            for path_index, path in enumerate(active_paths):
                logger.info(f"Processing path {path_index + 1}/{len(active_paths)}: {path}")

                # Step 1. Check if the current path has reached an answer
                answer_reached_result = self.llm_interface.check_path_reach_answers(
                    question=self.query.natural_language,
                    formatted_triples=self._trace_to_string(path.trace)
                )

                if answer_reached_result.reached:
                    logger.info(
                        f"Answer reached for path {path_index + 1}/{len(active_paths)}: {path}. "
                        f"Answer(s): {answer_reached_result.answer}, "
                        f"Explanation: {answer_reached_result.explanation}"
                    )

                    answers.extend(answer_reached_result.answer)
                    return answers

                # Step 2. If not reached, expand the current path with smapler and LLM support
                # Step 2a. PATHFINDER: Which relations should we follow from here?
        return answers
