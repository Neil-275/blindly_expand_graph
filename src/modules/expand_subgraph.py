from src.utils import fuse_mean, fuse_rrf, get_top_k, extract_numbers, extract_strings
from src.modules.shared import SENTENCE_TRANSFORMER_MODEL
from src.kb_interface.freebase import freebase_interface

from collections import defaultdict
from loguru import logger
import numpy as np
import sentence_transformers.util
import torch


class SubgraphSampler:
    def __init__(
        self,
        n_ent: int, n_rel: int,
        homoEdges: list,
        edge_index: list[list],
        args=None,
        fuse_func=fuse_mean,
        use_sub_objectives_a=False,
        use_sub_objectives_b=False,
        GoG_simulation=False,
        GoG_args=None
    ):
        # not using homoEdges currently
        self.args = args
        # self.model = ExpandSubgraph.model.to(args.device)
        self.model = SENTENCE_TRANSFORMER_MODEL.to(args.device)
        # self.util = util # SentenceTransformerUtil(model=self.model)
        self.k_rel_org = args.k_rel
        self.k_cands_org = args.k_cands
        self.k_rel = args.k_rel
        self.k_cands = args.k_cands
        self.k_decay = 0.9
        self.cands_lim = args.cands_lim
        self.GoG_simulation = GoG_simulation
        self.GoG_args = GoG_args
        if GoG_simulation:
            assert (GoG_args is not None), "Please provide GoG_args when GoG_simulation is True"
        self.orignal_edge_index = np.array(edge_index)
        self.edge_index = self.orignal_edge_index.copy()

        self._rel_embs_value = self.model.encode(
            list(
                rel_id
                for rel_id in edge_index[:, 1]
            ),
            convert_to_tensor=True
        ).to(self.args.device)

        self.rel_embs: dict[str, torch.Tensor] = {
            rel_id: self._rel_embs_value[idx]
            for idx, rel_id in enumerate(edge_index[:, 1])
        }

        self.adj = self.build_adjacency_list(self.edge_index)

        self.query: dict = None
        self.fuse_func = fuse_func
        self.subgraph_key = None
        self.answers_id = None
        self._name_cache: dict[str, str] = {}
        self.visited = set()

        if use_sub_objectives_a or use_sub_objectives_b:
            self.fuse_func = fuse_rrf
        self.use_sub_objectives_a = use_sub_objectives_a
        self.use_sub_objectives_b = use_sub_objectives_b
        self.sub_objectives = None

        self.query: dict = None
        self.query_emb: torch.Tensor = None
        self.start_entities: list[str] = None

    def build_adjacency_list(self):
        """
        Currently separating direct and inverse edges.
        Optimized to use defaultdict for better performance.
        """
        # print("Building adjacency list...")
        adjacency = defaultdict(list)
        for index, edges in enumerate(self.edge_index):
            head, rel, tail = edges
            adjacency[head].append((rel.item(), tail.item()))
        return adjacency

    def _id_to_name(self, ent_id: str) -> str:
        try:
            return self._name_cache[ent_id]
        except KeyError:
            name = freebase_interface.convert_id_to_name(ent_id)
            self._name_cache[ent_id] = name
            return name

    def _reset(self):
        self.subgraph_key = None
        self.visited = set()
        self.k_rels = self.k_rel_org
        self.k_cands = self.k_cands_org
        self._name_cache = {}

    # def assign_query(self, query: dict):
    #     self.query = query
    #     self.query_emb = self.model.encode(
    #         query['question'],
    #         convert_to_tensor=True
    #     ).to(self.args.device)

    def compare_relation_query_and_return_topK(
        self,
        triples: list[list[str]],
        fuse_func=fuse_mean
    ):
        """
        This function returns the top-k triples that are most relevant to the query
        based on the relation embeddings and the query embedding.
        """
        # Step 1. Get the unique relation ids and their corresponding embeddings
        triples = np.array(triples)

        unique_relation_mids = np.unique(triples[:, 1])
        relation_embeddings = [
            self.rel_embs[rel_id]
            for rel_id in unique_relation_mids
        ]

        # Step 2. Calculate the similarity scores between the query and the relation embeddings
        scores = sentence_transformers.util.dot_score(
            self.query_emb, relation_embeddings
        ).to('cpu')

        # Step 3. Get the top-k relation ids based on the similarity scores
        top_k_indices = get_top_k(
            scores=scores,
            k=min(self.k_rel, len(unique_relation_mids)),
            fuse_func=fuse_func
        )
        top_k_relation_mids = unique_relation_mids[top_k_indices]

        top_k_triples_mask = np.isin(triples[:, 1], top_k_relation_mids)
        top_k_triples = triples[top_k_triples_mask]

        return top_k_triples

    def remove_1hop_edges(self, triples: list[list[str]]):
        """
        This function removes the 1-hop edges from the given triples.
        """
        existing_doublets = np.array([
            [triple[0], triple[2]]
            for triple in self.edge_index
        ])
        start_entities = self.start_entities
        answers_mids = self.query.get("answers_id")

        golden_doublets = []
        for start_entity in start_entities:
            for answer_id in answers_mids:
                golden_doublets.append(np.array([start_entity, answer_id]))
                golden_doublets.append(np.array([answer_id, start_entity]))

        mask = np.ones(len(self.edge_index), dtype=bool)
        for doublet in golden_doublets:
            matches = np.where((existing_doublets == doublet).all(axis=1))[0]
            mask[matches] = False

        self.edge_index = self.edge_index[mask]

    def remove_golden_reasoning_paths(
        self,
    ):
        pass

    def remove_edges(self):
        pass

    def assign_query(self, query: dict):
        """
        This function assigns the query to subgraph sampler
        """

        natural_language_query = query.get('natural_language')
        if not natural_language_query:
            logger.warning("Query does not contain natural language form")
            raise ValueError("Query must contain 'natural_language' key.")

        # Extract all the neccessary information from the query
        query_embeddings: torch.Tensor = self.model.encode(
            natural_language_query,
            convert_to_tensor=True
        ).to(self.args.device)

        # This logic need to be changed
        # according to the dataset
        numbers = extract_numbers(query.get('query_structure', ()))

        # Extract the start entities from the query


        pass
