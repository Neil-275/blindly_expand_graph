system_prompt = """
You are an expert in natural language generation and translation from formal logic into conversational English.

## Task
Your task is to convert a given query in **logic form** into **three distinct and natural language versions** of the query. These versions must accurately capture the meaning of the logic form while sounding idiomatic and varied in their structure.

---
## Logic Form Specifications
The logic form is represented as a nested tuple string and uses the following symbols:
- **'e'**: Represents an **Entity** (a specific object, person, or place, e.g., 'Wesleyan University').
- **'r'**: Represents a **Relation** or **Predicate**.
    - If the relation starts with **"+"** or no symbol, it represents the **forward relation**.
    - If the relation starts with **"-"** (e.g., '-/people/person/place_of_birth'), the natural language query must express the **inverse of the relation**.
- **Implied Operator**: If no operator is explicitly declared between elements (e.g., two parenthesized logic statements), assume the **AND** operator (logical intersection).

---
## Input Format
The input will be a string representing the logic form, structured as a nested tuple.

**Example Input (Consistent with the defined format):**
Format: (e, (r,))
Values: ('Wesleyan University',
'+/education/educational_institution/students_graduates./education/education/major_field_of_study')

---
## Output Format
Your output must be a Python list of three strings, where each string is one of the natural language queries. Do not include any introductory or explanatory text, only the list.

**Example Output:**
["What are the major fields of study of students and graduates from Wesleyan University?",
"What did students who attended Wesleyan University major in?",
"What subjects did Wesleyan University students and graduates study as their major field?"]
"""
user_prompt_template = """
## Query to Process
Convert the following logic form into three natural language queries:
Format: {format}
Values: {query}
Output:
"""

split_prompt = """
You are an expert in natural language processing and logic parsing.

"""

extract_relation_prompt = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0.0 to 1.0 (the sum of the scores of %s relations is 1). Give your answer follow exactly to the format in the examples.
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
A: 1. {language.human_language.main_country (Score: 0.4)}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
2. {language.human_language.countries_spoken_in (Score: 0.3)}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
3. {base.rosetta.languoid.parent (Score: 0.2)}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.
<END>
Q: """

RELATION_EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge graph analyst specializing in relation extraction and relevance scoring. Your task is to identify the most relevant relations from a given set that would help answer a specific question about an entity.

INSTRUCTIONS:
1. Analyze the question to understand what information is being sought
2. Evaluate each relation based on its potential to contribute to answering the question
3. Select exactly the requested number of most relevant relations
4. Assign relevance scores (0.0 to 1.0) that sum to exactly 1.0
5. Provide clear reasoning for each relation's relevance

SCORING CRITERIA:
- Direct relevance: Relations that directly answer the question (higher scores: 0.4-0.6)
- Indirect relevance: Relations that provide supporting context (medium scores: 0.2-0.4)
- Minimal relevance: Relations that provide background information (lower scores: 0.1-0.2)

FORMAT REQUIREMENTS:
- List relations in order of relevance (highest score first)
- Use exact format: "{relation_name (Score: X.X)}: Explanation"
- Ensure scores sum to exactly 1.0
- End response with "<END>"
- Provide clear, concise explanations for each relation's relevance

EXAMPLE OF GOOD REASONING:
- Consider the question's intent and required information type
- Think about direct vs. indirect paths to the answer
- Evaluate how each relation contributes to building the complete answer
- Prioritize relations that lead to the most specific and accurate results"""

prompt_evaluate="""Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: {No}. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.
<END>
Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: {Yes}. Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {De Smet}.
<END>
Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: {No}. Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.
<END>
Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: {Yes}. Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.
<END>
Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: {No}. Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.
<END>
"""