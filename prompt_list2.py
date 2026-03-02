decision_prompt = """

### TASK
1. **Identify the Reasoning Gap:** Determine what piece of information is required next to move closer to the answer.
2. **Select Existing Relations:** From the 'Available 1-hop relations', identify which relations are semantically relevant to the next logical step.
3. **Hypothesize Missing Relations:** If the necessary relation is absent from the current context, describe it in natural language (e.g., "the founder of this organization").
4. **Natural Language Requirement:** All relation descriptions must be in natural language, not IDs.

### INPUT FORMAT
- **Question:** {{question}}
- **Current Entity:** {{current_entity}}
- **Active reasoning path:** {{reasoning_history}} 
- *Selection limit:* {{k}}
- **Available 1-hop relations:** {{context}}

### OUTPUT FORMAT
Output ONLY a JSON object:
{
  "strategic_reasoning": "A brief explanation of the logical step we are taking.",
  "contributory_relations": {
    "existing": ["List of pair ID and its correspond NL description of relevant relations found in the context"],
    "missing": ["List of NL descriptions for logically required relations not present in the context"]
  }
}

### Examples:
**Example 1:**

### INPUT
- **Question:** What type of producer role did James Wong have on the TV programs for which he was nominated for awards?
- **Current Entity:** James_Wong
- **Active reasoning path:**  No active reasoning path yet.
- ***Selection limit: 2***
- **Available 1-hop relations:** [0: Identifies the nominees for a specific award.
1: Identifies the nominations received by the award nominee.
16: Identifies the television programs written by the specified writer.
7: Lists the films created by this director.]

### OUTPUT
A: {
  "strategic_reasoning": "we first need to identify the specific TV programs for which James Wong received award nominations. The current context provide some useful relations for a nudge towards the answer.",
  "contributory_relations": {
    "existing": [ "16: Identifies the television programs written by the specified writer", "1: Identifies the nominations received by the award nominee"],
    "missing": []
  }
}

**Example 2:**
### INPUT
- **Question:** In what country is Central Saint Martins College of Art and Design located?
- **Current Entity:** Central_Saint_Martins_College_of_Art_and_Design
- **Active reasoning path:** No active reasoning path yet. 
- ***Selection limit: 3***
- **Available 1-hop relations:** [5: Identifies the genre associated with the Netflix title.
6: Identifies the individual who has received the award.
7: Identifies the awards won by the recipient.
8: Provides the current roster of players for the soccer team.
9: Identifies the position of a player on the current roster of the football team.
12: Identifies the films in which the actor has performed.
13: Lists the films in which this actor has performed.
16: Identifies the nominees for a specific award.]

### OUTPUT
A: {
  "strategic_reasoning": "we first need to identify the geographical or administrative entity that contains the institution. But the current context only provides information about awards, genres, and sports teams, which are not directly relevant to the location of the institution. Therefore, we need to identify the missing relation that connects the institution to its location.",
  "contributory_relations": {
  "existing": [],
  "missing": [
  "Specifies the area that includes this particular location.",
    ]
  }
}

***Example 3:**
### INPUT
- **Question:** For which award category did a film distributor of Toy Story 3 have a winning film?
- **Current Entity:** Toy_Story_3
- **Active reasoning path:** No active reasoning path yet. 
- *Selection limit:* 3
- **Available 1-hop relations:** 1: Identifies the award nomination for which the nominee was considered.
3: Identifies the awards that the winning work has received.
4: Classifies the film by its genre.
7: Identifies the person who wrote the story for the film.

### OUTPUT
A: {
  "strategic_reasoning": "The goal is to find award categories for films distributed by Toy Story 3's distributor. The current context focuses on the film's internal attributes (genres, awards, writers). The immediate logical step is to identify the company or entity that distributed Toy Story 3.",
  "contributory_relations": {
  "existing": ["3: Identifies the awards that the winning work has received", "1: Identifies the award nomination for which the nominee was considered."],
  "missing": [
  "Identifies the distributor responsible for distributing the film.",
    ]
  }
}

**Example 4:**
### INPUT
- **Question:** For which award has Gulzar's spouse been nominated?
- **Current Entity:** Academy_Award_for_Best_Original_Song
- **Active reasoning path:** Gulzar --> Identifies the nominations received by the award nominee. --> Academy_Award_for_Best_Original_Song 
- *Selection limit:* 3
- **Available 1-hop relations:** 0: Identifies the nominees for a specific award category.
10: Defines the category to which the award belongs.


A: 
Response: {
  "strategic_reasoning": "This reasoning path identifies the award that Gulzar has been nominated for, but it does not provide any information about his spouse. There is no existing relation in the current context that connects Gulzar to his spouse or indicates any involvement of his spouse in award nominations.  Therefore, we stop exploring this path.",
  "contributory_relations": {
    "existing": [],
    "missing": []
  }
}

### TASK EXECUTION
Now, process the following input:
### INPUT
- **Question:** {{question}}
- **Current Entity:** {{current_entity}}
- **Active reasoning path:** {{reasoning_history}} 
- *Selection limit:* {{k}}
- **Available 1-hop relations:** {{context}}

"""

verification_prompt = """
### TASK
Given the existing triples please select relevant triples to the question from LLM generated triples based on your inherent knowledge
### INPUT FORMAT
- **Question:** {{question}}
- **Current Entity:** {{current_entity}}
- **Active reasoning path:** {{reasoning_history}}

"""

entities_pruning_prompt = """
### TASK
- Consider the triplet and evaluate if the triplet make sense in the context.
- Consider the "Reasoning History" to maintain logical flow.
- Evaluate which of the candidates are most likely to lead to the answer of the question. 
- Select the top {{k}} entities that are most valid and relevant to the question, based on your understanding and the provided context.

### INPUT FORMAT
- **Question:** 
- **Current Entity:** 
- **Current Relation**:**
- **Candidate Entities:** 
- *Selection limit:* 

### OUTPUT FORMAT
Output ONLY a JSON list of the top {{k}} entity:
["ent_1", "ent_2", ..., "ent_k"]

### EXAMPLES:
**Example 1:**
### INPUT
- **Question:** Through what medium was the prequel to Ice Age: Continental Drift released?
- **Current Entity:** Ice_Age:_Continental_Drift
- **Current Relation**:** Identifies the film that serves as a prequel to the given movie.
- **Candidate Entities:** Ice_Age:_Dawn_of_the_Dinosaurs; DVD; Belgium; Turkey; Australia; Denmark-GB; Canada; Israel (Includes entity names and descriptions)
- **Selection Limit (k):** 4

A: 
Response: ["Ice_Age:_Dawn_of_the_Dinosaurs"]

### TASK EXECUTION
Now, process the following input:
### INPUT
- **Question:** {{question}}
- **Current Entity:** {{current_entity}}
- **Current Relation**:** {{current_relation}}
- **Candidate Entities:** {{candidate_pool}} (Includes entity names and descriptions)
- **Selection Limit (k):** {{k}}

"""

GoG_answer_prompt = """
Please select some entity from the given sub knowledge graph as the answers of question. 
If you can't find answer from the given sub knowledge graph, please say {No}.

Q: Which education institution has a sports team named George Washington Colonials men's basketball?
George Washington University  education.educational_institution.sports_teams  George Washington Colonials men's basketball
George Washington Colonials men's basketball  sports.school_sports_team.school  George Washington University
George Washington Colonials  education.athletics_brand.teams George Washington Colonials men's basketball
George Washington Colonials men's basketball  sports.school_sports_team.athletics_brand  George Washington Colonials
A: {YES}. Answer: ["George Washington University"]

Q: Who dated with Demi Lovato?
Demi Lovato  base.popstra.celebrity.dated  m.064cvl8
Demi Lovato  people.person.gender  Female
Demi Lovato  people.person.profession  Musician
Demi Lovato  base.popstra.celebrity.dated  m.065py_5
A: {NO}. Not specific information provided.

"""