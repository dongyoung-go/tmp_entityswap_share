# Trace Entity Swapper

Replaces entities in reasoning traces with alternative entities from Wikidata while preserving entities mentioned in the question.

## Installation

```bash
pip install requests tqdm spacy
python -m spacy download en_core_web_sm
```
## How It Works

### 1. Entity Selection for Substitution

The swapper identifies entities to replace by:

1. **Parsing triplets** from the trace using special tokens (`<|db_entity|>`, `<|db_relationship|>`, `<|db_return|>`, `<|db_end|>`)
2. **Extracting question entities** using spaCy NER (named entities, noun chunks, and significant tokens)
3. **Selecting return values** from triplets that are NOT in the question entities as candidates for substitution

### 2. Finding Similar Entities

For each entity to be replaced, the system:

#### Step 1: Entity Matching via Edit Distance
- Searches Wikidata for entities matching the label using edit distance similarity
- Uses `WikidataObjectMatcher` with configurable similarity threshold (default: 0.7)
- Selects the best match based on label similarity score

#### Step 2: Type Extraction
- Retrieves entity types using Wikidata property P31 ("instance of")
- Example: "Christopher Nolan" → "human"

#### Step 3: Alternative Entity Discovery
- Uses `EntityFinder` to locate entities with the same types
- Filters out:
  - The original entity
  - Already-used alternative entities (to ensure variety)
  - Entities already present in the trace
- Returns up to `num_alternatives` unique entities

### 3. Entity Label Reconstruction

When an entity contains question entities, the swapper preserve that part to make the chain reasonable:

1. **Splits** the label into preserved (question) and searchable parts
   - Example: "Inception director" where "Inception" is in the question
   - Preserved: "Inception", Searchable: "director"

2. **Finds alternatives** for only the searchable part

3. **Reconstructs** the full label preserving the question entity
   - Example: "director" → "engineer" becomes "Inception engineer"
   - Maintains word order from the original

### 4. Applying Swaps to Traces

The swapper applies entity substitutions using:

1. **Robust pattern matching**
   - Case-insensitive matching with word boundaries
   - Handles multiple spaces and special characters
   - Uses regex for precise whole-word replacement

2. **Capitalization preservation**
   - Detects original capitalization pattern (ALL CAPS, Title Case, lowercase)
   - Applies the same pattern to replacement text

3. **Global replacement**
   - Replaces all occurrences of the entity throughout the trace
   - Tracks replacement count per entity

## Configuration

Edit `config.json`:

```json
{
  "wikidata": {
    "endpoint": "https://query.wikidata.org/sparql",
    "user_agent": "TraceEntitySwapper/1.0",
    "rate_limit_delay": 0.5,
    "cache_dir": "./cache/wikidata",
    "timeout": 30
  },
  "generation": {
    "min_similarity_score": 0.7,
    "max_retries": 3
  }
}
```

## Usage

```python
from trace_entity_swapper import create_swapper_from_config

swapper = create_swapper_from_config('config.json')
results = swapper.swap_entities_in_trace(
    question="Which song, released on December 7, 2008 was co-written by Liz Rose?",
    trace="<thinking> The song released on December 7, 2008, that was co-written by Liz Rose is <|db_entity|> Liz Rose <|db_relationship|> co-written <|db_return|>White Horse, Teardrops on My Guitar, twenty of Taylor Swift's officially-released songs and singles, You Belong with Me<|db_end|> White Horse. </thinking>\n<answer> White Horse</answer>",
    num_alternatives=3
)
```

## Output Format

```json
[
  {
    "original_question": "Which song, released on December 7, 2008 was co-written by Liz Rose?",
    "original_trace": "...",
    "modified_trace": "<thinking> The song released on December 7, 2008, that was co-written by Liz Rose is <|db_entity|> Liz Rose <|db_relationship|> co-written <|db_return|>Shooting Star, Gangnam Style, twenty of Taylor Swift's officially-released songs and singles, Symphony No. 3<|db_end|> Shooting Star. </thinking>\n<answer> Shooting Star</answer>",
    "swaps": [
      {
        "original": "White Horse",
        "alternative": "Shooting Star",
        "occurrences": 3},
        {"original": "Teardrops on My Guitar",
        "alternative": "Gangnam Style",
        "occurrences": 1},
        {"original": "You Belong with Me",
        "alternative": "Symphony No. 3",
        "occurrences": 1
      }
    ]
  }
]
```
