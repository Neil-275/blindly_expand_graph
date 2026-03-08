import os
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Union
import urllib.error


sparql = SPARQLWrapper(os.environ['SPARQLPATH'])
sparql.setReturnFormat(JSON)
ns_prefix = "http://rdf.freebase.com/ns/"


class FreebaseInterface:
    def __init__(self, **kwargs):
        pass

    def execurte_sparql(self, sparql_query):
        sparql.setQuery(sparql_query)
        results = sparql.query().convert()
        return results["results"]["bindings"]

    def replace_relation_prefix(self, relations):
        return [
            relation["relation"]["value"].replace("http://rdf.freebase.com/ns/", "")
            for relation in relations
        ]

    def replace_entities_prefix(self, entities):
        return [
            entity["entity"]["value"].replace("http://rdf.freebase.com/ns/", "")
            if entity['entity']['type'] == 'uri' else "@." + entity['entity']['value']
            for entity in entities
        ]

    def _remove_ns(self, entity_id: str):
        if entity_id.startswith("ns:"):
            entity_id = entity_id[3:]
        return entity_id

    def _add_ns(self, entity_id: str):
        if not entity_id.startswith("ns:"):
            entity_id = "ns:" + entity_id
        return entity_id

    def get_tail_entity(self, entity_id, relation):
        sparql_pattern = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?entity
        WHERE {{
            ns:{head} ns:{relation} ?entity .
        }}
        """
        sparql_text = sparql_pattern.format(head=entity_id, relation=relation)
        entities = self.execurte_sparql(sparql_text)

        entities = self.replace_entities_prefix(entities)
        entity_ids = [
            entity for entity in entities
            if entity.startswith("m.") or entity.startswith("@.")
        ]

        return entity_ids

    def get_head_entity(self, entity_id, relation):
        sparql_pattern = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?entity
        WHERE {{
            ?entity ns:{relation} ns:{tail}  .
        }}
        """
        sparql_text = sparql_pattern.format(tail=entity_id, relation=relation)
        entities: list[str] = self.execurte_sparql(sparql_text)
        entities = self.replace_entities_prefix(entities)
        entity_ids = [
            entity
            for entity in entities
            if entity.startswith("m.") or entity.startswith("@.")
        ]

        return entity_ids

    def get_out_relations(self, entity_id):
        sparql_pattern = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?relation
        WHERE {{
            ns:{head} ?relation ?x .
        }}
        """
        sparql_text = sparql_pattern.format(head=entity_id)
        relations = self.execurte_sparql(sparql_text)

        relations = self.replace_relation_prefix(relations)

        return relations

    def get_in_relations(self, entity_id):
        sparql_pattern = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?relation
        WHERE {{
            ?x ?relation ns:{tail} .
        }}"""
        sparql_text = sparql_pattern.format(tail=entity_id)
        relations = self.execurte_sparql(sparql_text)

        relations = self.replace_relation_prefix(relations)

        return relations

    def convert_id_to_name(self, entity_id):
        entity_id = self._remove_ns(entity_id)

        sparql_id = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?tailEntity
        WHERE {{
            {{
                ?entity ns:type.object.name ?tailEntity .
                FILTER(?entity = ns:%s)
            }}
            UNION
            {{
                ?entity ns:common.topic.alias ?tailEntity .
                FILTER(?entity = ns:%s)
            }}
        }}
        """
        sparql_query = sparql_id % (entity_id, entity_id)

        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        if len(results["results"]["bindings"]) == 0:
            return entity_id
        else:
            return results["results"]["bindings"][0]["tailEntity"]["value"]

    def convert_name_to_id(self, label: str):
        sparql_id = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?entity
        WHERE {{
            {{
                ?entity ns:type.object.name "%s"@en .
            }}
            UNION
            {{
                ?entity ns:common.topic.alias "%s"@en .
            }}
        }}
        """
        sparql_query = sparql_id % (label, label)

        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        if len(results["results"]["bindings"]) == 0:
            return label
        else:
            return results["results"]["bindings"][0]["entity"]["value"].replace(ns_prefix, "")

    def pre_filter_relations(self, relations: List[str]):
        ignored_relations = [
            "type.object.type",
            "type.object.name",
        ]
        filtered_relations = []
        for relation in relations:
            if (
                relation in ignored_relations
                or relation.startswith("freebase.")
                or relation.startswith("common.")
                or relation.startswith("kg.")
            ):
                continue
            else:
                filtered_relations.append(relation)
        return filtered_relations

    def get_1hop_triples(self, entity_ids: Union[str, List]):
        if type(entity_ids) is str:
            entity_ids = [entity_ids]

        entity_ids = [
            self._add_ns(entity_id)
            for entity_id in entity_ids
        ]
        triples = set()

        query = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?mid ?subject ?predicate ?object WHERE {{
            VALUES ?mid {{ {entity_ids} }}
            {{ ?subject ?predicate ?mid }}
            UNION
            {{ ?mid ?predicate ?object }}
            FILTER regex(?predicate, "http://rdf.freebase.com/ns/")
        }}
        """.format(
            entity_ids=" ".join(entity_ids)
        )

        sparql.setQuery(query)
        try:
            results = sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        for result in results["results"]["bindings"]:
            mid = result["mid"]["value"].replace(ns_prefix, "")
            predicate = result["predicate"]["value"].replace(ns_prefix, "")
            if "subject" in result:
                subject = result["subject"]["value"].replace(ns_prefix, "")
                triples.add((subject, predicate, mid))
            elif "object" in result:
                object = result["object"]["value"].replace(ns_prefix, "")
                triples.add((mid, predicate, object))

        relations = list(set([triple[1] for triple in triples]))
        relations = self.pre_filter_relations(relations)

        triples = [list(triple) for triple in triples if triple[1] in relations]

        return triples, relations


# Singleton instance of FreebaseInterface
freebase_interface = FreebaseInterface()
