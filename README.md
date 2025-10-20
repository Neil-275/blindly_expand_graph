# Project Setup Instructions

## Data Construction

### Required Data Files

Place the following files in `/knowledge_graph/KG_data/FB15k-237-betae/`:

1. **Query and Answer Files**:
   - `train-queries.pkl` - Training queries
   - `train-answers.pkl` - Training answers
   - `valid-easy-answers.pkl` - Validation answers
   - `test-easy-answers.pkl` - Test answers

2. **Knowledge Graph**:
   - `train.txt` - Training graph edges 

3. **Entity and Relation Mappings**:
   - `id2ent.pkl` - ID to entity mapping
   - `id2rel.pkl` - ID to relation mapping
   - `ent2id.pkl` - Entity to ID mapping
   - `FB15k_mid2name.txt` - Freebase MID to name mapping 

### Environment Setup

1. Create a `.env` file in the project root with necessary environment variables


### Running the Project

1. Ensure all data files are in the correct locations
2. Run the main script or Jupyter notebooks in the project directory