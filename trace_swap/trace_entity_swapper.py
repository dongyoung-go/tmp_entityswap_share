#!/usr/bin/env python3
"""
Trace Entity Swapper - Replace entities in traces with alternative entities from Wikidata.

This module:
1. Takes input question and input trace
2. Extracts triplets from the trace
3. For each triplet's return values (objects), finds matching entities in Wikidata
4. Gets entity types (P31 - instance of) from Wikidata
5. Finds alternative entities of the same type
6. Replaces all occurrences of the original entity in the trace with the alternative

Installation requirements documented in README.md
"""

import logging
import re
import sys
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import spacy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wikidata_matcher import WikidataObjectMatcher, EditDistanceSimilarity
from utils import WikidataClient, EntityFinder, Config

UNKNOWN = 'unknown'
# ============================================================================
# Logging Configuration
# ============================================================================

# Configure default logging for the module
# This will show INFO and WARNING logs even when imported as a module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ============================================================================
# TraceEntitySwapper
# ============================================================================

class TraceEntitySwapper:
    """
    Swap entities in traces with alternative entities from Wikidata.
    """

    def __init__(self,
                 wikidata_client: WikidataClient,
                 entity_finder: EntityFinder,
                 matcher: WikidataObjectMatcher,
                 min_similarity_threshold: float = 0.7):
        """
        Initialize TraceEntitySwapper.

        Args:
            wikidata_client: Client for Wikidata SPARQL queries
            entity_finder: Finder for alternative entities
            matcher: Matcher for finding entities by label with edit distance
            min_similarity_threshold: Minimum similarity score to consider a match
        """
        self.wikidata = wikidata_client
        self.entity_finder = entity_finder
        self.matcher = matcher
        self.min_similarity_threshold = min_similarity_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.question_entities = []  # Cache for question entities
        self.used_alternative_ids = set()  # Track used alternative entity IDs
        self.existing_ids = dict()  # Track existing entity IDs

        # Load spaCy NLP model for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            self.logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Please install it with: python -m spacy download en_core_web_sm"
            )
            raise

    def swap_entities_in_trace(self,
                               question: str,
                               trace: str,
                               num_alternatives: int = 1) -> List[Dict[str, Any]]:
        """
        Main method to swap entities in a trace.

        Args:
            question: Input question
            trace: Input trace containing triplets
            num_alternatives: Number of alternative entities to find per entity

        Returns:
            List of modified traces with entity swaps, each containing:
            {
                'original_question': str,
                'modified_trace': str,
                'swaps': List[Dict] with swap information
            }
        """
        self.logger.info(f"Processing question: {question}")

        # Reset used alternatives for this trace
        self.used_alternative_ids = set()
        self.existing_ids = dict()

        # Step 1: Extract entities from question to preserve them
        self.question_entities = self._extract_entities_from_question(question)
        self.logger.info(f"Extracted {len(self.question_entities)} entities from question: {self.question_entities}")

        # Step 2: Parse triplets from trace
        triplets = parse_triplets(trace)
        if not triplets:
            self.logger.warning("No triplets found in trace")
            return []

        self.logger.info(f"Found {len(triplets)} triplets")
        
        
        unique_values = list(set(sum([triple['return_values'] for triple in triplets], [])))
        print('unique_values:', unique_values)
        for label in unique_values:
            q_id = self._label_to_id(label)
            if q_id:
                self.existing_ids[label] = {'q_id' : q_id,
                                            'tested': False,
                                            }
        print('self.existing_ids:', self.existing_ids)
        # Step 3-6: Process each triplet and collect swaps
        all_swaps = []

        for i, triplet in enumerate(triplets):
            self.logger.info(f"Processing triplet {i+1}/{len(triplets)}")
            self.logger.info(f"  Entity: {triplet['entity']}")
            self.logger.info(f"  Relationship: {triplet['relationship']}")
            self.logger.info(f"  Return values: {triplet['return_values']}")

            # Step 4: Extract value (last element in triplet = return_values)
            return_values = triplet['return_values']

            # Step 5-6: For each return value, find alternatives
            for value in return_values:
                if value == UNKNOWN:
                    self.logger.info(f"Skipping unknown entity: {value}")
                    continue
                if self.existing_ids.get(value, {}).get('tested', False):
                    self.logger.info(f"Skipping already tested entity: {value}")
                    continue
                swaps = self._find_alternative_entities(
                    original_label=value,
                    num_alternatives=num_alternatives
                )

                if swaps:
                    all_swaps.extend(swaps)

        # Step 7: Generate modified traces by applying swaps
        if not all_swaps:
            self.logger.warning("No entity swaps found")
            return []

        # Group swaps by original entity to avoid conflicts
        swap_groups = self._group_swaps_by_original(all_swaps)

        # Generate one modified trace per swap combination
        results = []
        for swap_group in swap_groups[:num_alternatives]:  # Limit to num_alternatives results
            modified_trace, applied_swaps = self._apply_swaps_to_trace(trace, swap_group)
            results.append({
                'original_question': question,
                'original_trace': trace,
                'modified_trace': modified_trace,
                'swaps': applied_swaps
            })

        return results

    def _extract_entities_from_question(self, question: str) -> List[str]:
        """
        Extract entities from question using spaCy NER.

        Uses spaCy's named entity recognition to identify entities in the question.
        Extracts entities of common types: PERSON, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, etc.

        Args:
            question: Input question

        Returns:
            List of entity labels found in the question
        """
        entities = []

        # Process question with spaCy
        doc = self.nlp(question)

        # Extract all named entities
        for ent in doc.ents:
            entities.append(ent.text.strip())
        for chunk in doc.noun_chunks:
            entities.append(chunk.text.strip())
        ALLOWED_POS = {"NOUN", "PROPN", "ADJ", "VERB"}
        for token in doc:
            if token.pos_ in ALLOWED_POS and not token.is_stop and token.is_alpha:
                entities.append(token.text.strip())

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)

        return unique_entities

    def _label_to_id(self, search_label: str) -> Optional[str]:
        entity_candidates = self.matcher.search_entity_by_label(
            label=search_label,
            limit=5
        )

        if not entity_candidates:
            self.logger.warning(f"No entity found in Wikidata for: {search_label}")
            return ''

        # Take the best match (first result)
        best_match = entity_candidates[0]

        # Check similarity using edit distance
        similarity_measure = EditDistanceSimilarity()
        similarity = similarity_measure.calculate(search_label, best_match['label'])

        if similarity < self.min_similarity_threshold:
            self.logger.warning(
                f"Best match '{best_match['label']}' has low similarity ({similarity:.4f}) "
                f"to search label '{search_label}'"
            )
            return ''

        self.logger.info(
            f"Found entity: {best_match['label']} ({best_match['id']}) "
            f"with similarity {similarity:.4f}"
        )
        return best_match['id']

    def _find_alternative_entities(self,
                                   original_label: str,
                                   num_alternatives: int = 1) -> List[Dict[str, Any]]:
        """
        Find alternative entities for a given label.

        If the label contains a question entity, only replace the non-question part.
        For example, if value is "XXX YYY" and "YYY" is in question entities,
        only search for "XXX" and preserve "YYY".

        Args:
            original_label: Original entity label
            num_alternatives: Number of alternatives to find

        Returns:
            List of swap dictionaries containing original and alternative entity info
        """
        # Check if any question entity is part of this value
        if original_label in self.existing_ids:
            self.existing_ids[original_label]['tested'] = True
        preserved_part, searchable_part = self._split_by_question_entity(original_label)

        if preserved_part:
            self.logger.info(
                f"Value '{original_label}' contains question entity '{preserved_part}'. "
                f"Will only search for '{searchable_part}'"
            )
            search_label = searchable_part
        else:
            search_label = original_label

        # Step 4a: Search for entity in Wikidata using edit distance
        entity_id = self._label_to_id(search_label)

        # Step 4b: Get entity types (P31 - instance of)
        entity_types = self.wikidata.get_entity_types(entity_id)

        if not entity_types:
            self.logger.warning(f"No types found for entity: {entity_id}")
            return []

        self.logger.info(f"Entity types: {entity_types}")

        # Step 5: Find alternative entities of the same type
        alternative_entity_ids = self.entity_finder.find_alternatives(
            original_entity_id=entity_id,
            required_property=None,
            num_alternatives=num_alternatives,
            exclude_entities=list(self.used_alternative_ids) + [_value['q_id'] for _value in self.existing_ids.values()],
        )

        if not alternative_entity_ids:
            self.logger.warning(f"No alternatives found for entity: {entity_id}")
            return []

        # Get labels for alternative entities
        swaps = []
        for alt_id in alternative_entity_ids:
            alt_data = self.wikidata.get_entity_label_and_aliases(alt_id)
            alt_label = alt_data['label']

            if alt_label is None:
                self.logger.warning(f"No label found for alternative entity: {alt_id}")
                continue

            # If we preserved part of the original, reconstruct the full label
            if preserved_part:
                # Determine the order: was preserved part at start or end?
                if original_label.startswith(preserved_part):
                    full_alt_label = preserved_part + " " + alt_label
                else:
                    full_alt_label = alt_label + " " + preserved_part
            else:
                full_alt_label = alt_label

            swaps.append({
                'original_label': original_label,
                'original_id': entity_id,
                'alternative_label': full_alt_label,
                'alternative_id': alt_id,
                'types': entity_types,
                'preserved_part': preserved_part if preserved_part else None
            })

            # Track this alternative to avoid reuse
            self.used_alternative_ids.add(alt_id)

            self.logger.info(
                f"Found alternative: {full_alt_label} ({alt_id})"
            )

        return swaps

    def _split_by_question_entity(self, value: str) -> Tuple[Optional[str], str]:
        """
        Split value into preserved (question entity) and searchable parts.

        Args:
            value: Original value string

        Returns:
            Tuple of (preserved_part, searchable_part)
            If no question entity found, returns (None, original_value)
        """
        value_lower = value.lower()

        # Check each question entity
        for q_entity in self.question_entities:
            q_entity_lower = q_entity.lower()

            # Check if question entity is in value (case-insensitive)
            if q_entity_lower in value_lower:
                # Find the position
                pos = value_lower.find(q_entity_lower)

                # Extract the actual case-preserved question entity from value
                preserved = value[pos:pos + len(q_entity)]

                # Get the remaining part
                if pos == 0:
                    # Question entity at start
                    searchable = value[len(q_entity):].strip()
                else:
                    # Question entity at end (or middle, but we'll assume end)
                    searchable = value[:pos].strip()

                return preserved, searchable

        # No question entity found
        return None, value

    def _group_swaps_by_original(self,
                                 swaps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group swaps by original entity to create non-conflicting swap sets.

        Ensures no duplicate swap groups by tracking unique combinations of alternative entity IDs.

        Args:
            swaps: List of swap dictionaries

        Returns:
            List of unique swap groups, where each group contains one swap per unique original entity
        """
        # Group by original entity
        by_original = {}
        for swap in swaps:
            original = swap['original_label']
            if original not in by_original:
                by_original[original] = []
            by_original[original].append(swap)

        if not by_original:
            return []

        # Create unique swap groups
        swap_groups = []
        seen_combinations = set()  # Track seen alternative ID combinations

        # Get all originals
        originals = list(by_original.keys())

        # For each alternative of the first original, create a group
        first_original = originals[0]
        for swap in by_original[first_original]:
            group = [swap]
            # Add first alternative for other originals
            for original in originals[1:]:
                if by_original[original]:
                    group.append(by_original[original][0])

            # Create a unique identifier for this combination based on alternative IDs
            # Sort by original_label to ensure consistent ordering
            sorted_group = sorted(group, key=lambda x: x['original_label'])
            combination_key = tuple(s['alternative_id'] for s in sorted_group)

            # Only add if this combination hasn't been seen before
            if combination_key not in seen_combinations:
                seen_combinations.add(combination_key)
                swap_groups.append(group)

        return swap_groups

    def _apply_swaps_to_trace(self,
                             trace: str,
                             swaps: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Apply entity swaps to the entire trace with robust matching.

        Uses case-insensitive matching with word boundaries to avoid partial matches.
        Preserves the original capitalization pattern when possible.

        Args:
            trace: Original trace
            swaps: List of swap dictionaries

        Returns:
            Tuple of (modified_trace, applied_swaps)
        """
        modified_trace = trace
        applied_swaps = []

        for swap in swaps:
            original = swap['original_label']
            alternative = swap['alternative_label']

            # Create regex pattern for case-insensitive, whole-word matching
            # Use word boundaries, but also handle punctuation
            pattern = self._create_robust_pattern(original)

            # Find all matches (case-insensitive)
            matches = list(re.finditer(pattern, modified_trace, re.IGNORECASE))

            if not matches:
                self.logger.warning(f"Original entity '{original}' not found in trace")
                continue

            # Replace all occurrences, preserving capitalization where appropriate
            count = len(matches)

            # Replace from end to start to preserve match positions
            for match in reversed(matches):
                matched_text = match.group(0)

                # Preserve capitalization pattern if possible
                replacement = self._preserve_capitalization(matched_text, alternative)

                # Replace this specific occurrence
                start, end = match.span()
                modified_trace = modified_trace[:start] + replacement + modified_trace[end:]

            applied_swaps.append({
                'original': original,
                'alternative': alternative,
                'occurrences': count
            })

            self.logger.info(
                f"Replaced '{original}' with '{alternative}' ({count} occurrences)"
            )

        return modified_trace, applied_swaps

    def _create_robust_pattern(self, entity_name: str) -> str:
        """
        Create a robust regex pattern for entity matching.

        Handles:
        - Word boundaries
        - Special regex characters
        - Multiple spaces

        Args:
            entity_name: Entity name to create pattern for

        Returns:
            Regex pattern string
        """
        # Escape special regex characters
        escaped = re.escape(entity_name)

        # Replace escaped spaces with flexible space pattern (handles multiple spaces)
        escaped = escaped.replace(r'\ ', r'\s+')

        # Add word boundaries
        # Use \b for alphanumeric boundaries, but be flexible with punctuation
        pattern = r'\b' + escaped + r'\b'

        return pattern

    def _preserve_capitalization(self, original_text: str, replacement: str) -> str:
        """
        Preserve capitalization pattern from original text in replacement.

        Handles common patterns:
        - ALL CAPS
        - Title Case
        - lowercase
        - First word capitalized

        Args:
            original_text: Original matched text with its capitalization
            replacement: Replacement text

        Returns:
            Replacement text with preserved capitalization pattern
        """
        # If original is all uppercase, make replacement all uppercase
        if original_text.isupper():
            return replacement.upper()

        # If original is all lowercase, make replacement all lowercase
        if original_text.islower():
            return replacement.lower()

        # If original is title case (each word capitalized)
        if original_text.istitle():
            return replacement.title()

        # If just first character is capitalized
        if original_text[0].isupper() and original_text[1:].islower():
            return replacement.capitalize()

        # Default: use replacement as-is
        return replacement


# ============================================================================
# Utility Functions
# ============================================================================

def parse_triplets(text: str) -> List[Dict[str, Any]]:
    """
    Extract triplets from natural language text.
    
    Args:
        text: Text containing special tokens, <|db_entity|>, <|db_relationship|>, <|db_return|>, <|db_end|>.
        
    Returns:
        List of triplets. Each triplet is a dict with the following structure:
        {
            'entity': str,
            'relationship': str,
            'return_values': List[str]
        }
    """
    triplets = []
    
    # Regular expression pattern: match from <|db_entity|> to <|db_end|>
    pattern = r'<\|db_entity\|>\s*(.*?)\s*<\|db_relationship\|>\s*(.*?)\s*<\|db_return\|>\s*(.*?)\s*<\|db_end\|>'
    
    # Find all triplets
    matches = re.finditer(pattern, text)
    
    for match in matches:
        entity = match.group(1).strip()
        relationship = match.group(2).strip()
        return_str = match.group(3).strip()
        
        # Split return values by comma and strip whitespace
        return_values = [val.strip() for val in return_str.split(',')]
        
        triplets.append({
            'entity': entity,
            'relationship': relationship,
            'return_values': return_values
        })
    
    return triplets

def create_swapper_from_config(config_path: str) -> TraceEntitySwapper:
    """
    Create TraceEntitySwapper from configuration file.

    Args:
        config_path: Path to config JSON file

    Returns:
        Configured TraceEntitySwapper instance
    """
    config = Config.from_file(config_path)

    wikidata_client = WikidataClient(config)
    entity_finder = EntityFinder(wikidata_client)
    matcher = WikidataObjectMatcher(
        endpoint=config.wikidata_endpoint,
        user_agent=config.user_agent,
        rate_limit_delay=config.rate_limit_delay,
        timeout=config.timeout,
        similarity_measure=EditDistanceSimilarity()
    )

    return TraceEntitySwapper(
        wikidata_client=wikidata_client,
        entity_finder=entity_finder,
        matcher=matcher,
        min_similarity_threshold=config.min_similarity_score
    )


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Swap entities in traces with alternative entities from Wikidata'
    )
    parser.add_argument(
        '--question',
        type=str,
        required=True,
        help='Input question'
    )
    parser.add_argument(
        '--trace',
        type=str,
        required=True,
        help='Input trace containing triplets'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--num-alternatives',
        type=int,
        default=1,
        help='Number of alternative entities to find per entity'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (JSON format). If not provided, prints to stdout'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging (force=True to override module-level config)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Override the default INFO level set at module import
    )

    # Create swapper
    swapper = create_swapper_from_config(args.config)

    # Process trace
    results = swapper.swap_entities_in_trace(
        question=args.question,
        trace=args.trace,
        num_alternatives=args.num_alternatives
    )

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    else:
        print(json.dumps(results, indent=2))

    # Print summary
    print(f"\nProcessed {len(results)} alternative trace(s)")
    for i, result in enumerate(results, 1):
        print(f"\nAlternative {i}:")
        print(f"  Swaps applied: {len(result['swaps'])}")
        for swap in result['swaps']:
            print(f"    - {swap['original']} → {swap['alternative']} ({swap['occurrences']} occurrences)")


if __name__ == '__main__':
    main()
