#!/usr/bin/env python3
"""
Generate alternative counterfactual variations for MQuAKE dataset.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import Config, WikidataClient, EntityFinder


# ============================================================================
# ChainBuilder and AnswerTracer
# ============================================================================

class ChainBuilder:
    """Generate alternative fact chains"""

    def __init__(self, entity_finder: EntityFinder, wikidata_client: WikidataClient):
        self.entity_finder = entity_finder
        self.wikidata = wikidata_client
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _normalize_triples(instance: Dict) -> Tuple[List, List]:
        """
        Normalize triples format to handle both nested and flat formats.

        Returns:
            Tuple of (triples, triples_labeled)
        """
        # Try nested format first (old format)
        # https://github.com/princeton-nlp/MQuAKE?tab=readme-ov-file#data-format
        if 'orig' in instance and isinstance(instance['orig'], dict):
            if 'triples' in instance['orig']:
                return instance['orig']['triples'], instance['orig']['triples_labeled']

        # Try flat format (remastered format)
        # https://huggingface.co/datasets/henryzhongsc/MQuAKE-Remastered
        if 'orig_triples' in instance:
            return instance['orig_triples'], instance['orig_triples_labeled']

        raise ValueError("Unable to find triples in either nested or flat format")

    def generate_alternative_chain(self, original_instance: Dict) -> Optional[Dict]:
        """
        Generate one alternative counterfactual chain

        Strategy: Replace entities in the edited chain with similar alternatives
        Handles multiple edits across different positions in the chain
        """
        triples, _ = self._normalize_triples(original_instance)
        edit_triples_idx = [idx for idx in range(len(triples))] # change every original triple

        if not edit_triples_idx:
            return None

        # Collect entities to exclude (from original and edited chains)
        exclude_entities = [t[2] for t in triples]

        # Build alternative entities for each edit position
        # Key: edit index, Value: list of alternative entity IDs
        edit_alternatives = {}

        for i, edit_idx in enumerate(edit_triples_idx):

            edit_triple = triples[i]
            orig_obj_id = edit_triple[2]  # We only need the object from the edit triple

            # Determine required property for chain continuity
            required_property = None

            # Check if there's a next triple that needs to connect
            if edit_idx + 1 < len(triples):
                # Get the relation from the next triple
                next_rel_id = triples[edit_idx + 1][1]
                required_property = next_rel_id

            # Find alternative entities for this edit position
            alternatives = self.entity_finder.find_alternatives(
                original_entity_id=orig_obj_id,
                required_property=required_property,
                num_alternatives=5,
                exclude_entities=exclude_entities
            )

            if not alternatives:
                self.logger.warning(f"No alternatives found for edit position {edit_idx}, entity {orig_obj_id}")
                return None

            edit_alternatives[i] = alternatives

        # Try to build a complete chain with alternative entities
        # For simplicity, we'll try combinations starting with the first alternative of the first edit
        result = self._try_build_chain_multi_edit(
            triples=triples,
            edit_triples_idx=edit_triples_idx,
            edit_alternatives=edit_alternatives
        )

        # Include the edit_triples_idx in the result so it can be used in instance building
        if result:
            result['edit_triples_idx'] = edit_triples_idx

        return result

    def _try_build_chain_multi_edit(self,
                                     triples: List[List[str]],
                                     edit_triples_idx: List[int],
                                     edit_alternatives: Dict[int, List[str]]) -> Optional[Dict]:
        """
        Try to build a complete chain with multiple edits

        Args:
            triples: Original triple chain
            edit_triples_idx: Indices of edited positions in the original chain
            edit_alternatives: Map of edit index to list of alternative entity IDs

        Returns:
            Dict with new_triples and final_answer_id, or None if chain cannot be built
        """
        # Try combinations of alternatives for each edit position
        # For now, we'll try each alternative for the first edit, then for each subsequent edit
        from itertools import product

        # Create list of alternatives for each edit, limiting combinations
        alternative_combinations = []
        for i in range(len(edit_triples_idx)):
            if i in edit_alternatives:
                # Limit to first 3 alternatives per edit to avoid too many combinations
                alternative_combinations.append(edit_alternatives[i][:3])
            else:
                # If no alternatives found for this edit, use the original object from the original triple
                edit_idx = edit_triples_idx[i]
                alternative_combinations.append([triples[edit_idx][2]])

        # Try each combination
        for combination in product(*alternative_combinations):
            result = self._try_build_single_chain(
                triples=triples,
                edit_triples_idx=edit_triples_idx,
                alternative_entities=combination
            )
            if result:
                return result

        return None

    def _try_build_single_chain(self,
                                triples: List[List[str]],
                                edit_triples_idx: List[int],
                                alternative_entities: Tuple[str, ...]) -> Optional[Dict]:
        """
        Try to build a single chain with specific alternative entities for each edit

        Args:
            triples: Original triple chain
            edit_triples_idx: Indices of edited positions
            alternative_entities: Tuple of alternative entity IDs for each edit position

        Returns:
            Dict with new_triples and final_answer_id, or None if chain breaks
        """
        new_triples = []
        edit_map = {idx: i for i, idx in enumerate(edit_triples_idx)}

        for i in range(len(triples)):
            if i in edit_map:
                # This position is edited
                edit_index = edit_map[i]
                original_triple = triples[i]  # Get from original triples

                # Get subject from original triple (or previous triple's object for chaining)
                if i == 0:
                    subj_id = original_triple[0]
                else:
                    subj_id = new_triples[i-1][2]

                rel_id = original_triple[1]  # Use relation from original triple
                # Use the alternative entity for this edit
                obj_id = alternative_entities[edit_index]

                new_triples.append([subj_id, rel_id, obj_id])
            else:
                # This position is not edited, follow chain from previous triple
                if i == 0:
                    # First triple, use original
                    new_triples.append(triples[i])
                else:
                    # Get subject from previous triple's object
                    subj_id = new_triples[i-1][2]
                    rel_id = triples[i][1]

                    # Query Wikidata for the next entity
                    next_entities = self.wikidata.get_entity_property(subj_id, rel_id)

                    if not next_entities:
                        # Chain breaks
                        return None

                    # Take the first result
                    obj_id = next_entities[0]
                    new_triples.append([subj_id, rel_id, obj_id])

        # Verify chain length matches original
        if len(new_triples) != len(triples):
            return None

        # Final answer is the last object in chain
        final_answer_id = new_triples[-1][2]

        return {
            'new_triples': new_triples,
            'final_answer_id': final_answer_id
        }


class AnswerTracer:
    """Trace through fact chains to determine final answer"""

    def __init__(self, wikidata_client: WikidataClient):
        self.wikidata = wikidata_client

    def get_answer_with_aliases(self, answer_entity_id: str) -> Dict[str, Any]:
        """Get answer label and all aliases"""
        result = self.wikidata.get_entity_label_and_aliases(answer_entity_id)
        return {
            'str': result['label'] or answer_entity_id,
            'id': answer_entity_id,
            'aliases': result['aliases']
        }


# ============================================================================
# InstanceBuilder
# ============================================================================

class InstanceBuilder:
    """Build complete MQuAKE instances from alternative chains"""

    def __init__(self, answer_tracer: AnswerTracer, wikidata_client: WikidataClient,
                 question_templates: Dict, cloze_templates: Dict, relation_labels: Dict):
        self.answer_tracer = answer_tracer
        self.wikidata = wikidata_client
        self.question_templates = question_templates
        self.cloze_templates = cloze_templates
        self.relation_labels = relation_labels
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _normalize_target(target_data: Dict) -> Tuple[str, str]:
        """
        Normalize target format to handle both nested and flat formats.

        Args:
            target_data: Either {'str': ..., 'id': ...} or already at rewrite level

        Returns:
            Tuple of (str, id)
        """
        # Check if it's nested format
        if 'str' in target_data and 'id' in target_data:
            return target_data['str'], target_data['id']

        # Check if it's flat format (at the rewrite level, not nested)
        # This shouldn't happen in this function, but handle it just in case
        if 'target_new_str' in target_data and 'target_new_id' in target_data:
            return target_data['target_new_str'], target_data['target_new_id']

        raise ValueError(f"Unable to parse target data: {target_data}")

    def build_variation_instance(self, original_instance: Dict, chain_result: Dict,
                                 variation_id: int) -> Optional[Dict]:
        """Build complete instance matching MQuAKE schema"""
        # Use the actual edit_triples_idx from chain generation
        edit_triples_idx = chain_result.get('edit_triples_idx', [])

        # Normalize input format
        orig_triples, orig_triples_labeled = ChainBuilder._normalize_triples(original_instance)

        # Start with a copy of original instance structure (using flat format)
        instance = {
            'case_id': f"{original_instance['case_id']}_v{variation_id}",
            'requested_rewrite': [],
            'questions': original_instance['questions'],  # Keep same
            'answer': original_instance['answer'],  # Keep same (original answer)
            'answer_alias': original_instance['answer_alias'],  # Keep same
            'new_answer': None,
            'new_answer_alias': [],
            'single_hops': original_instance['single_hops'],  # Keep same
            'new_single_hops': [],
            # Use flat format for output
            'orig_triples': orig_triples,
            'orig_triples_labeled': orig_triples_labeled,
            'new_triples': [],
            'new_triples_labeled': [],
            'edit_triples': [], # leave empty following remastered format
            'edit_triples_idx': edit_triples_idx,
        }

        # Get new answer
        answer_data = self.answer_tracer.get_answer_with_aliases(chain_result['final_answer_id'])
        instance['new_answer'] = answer_data['str']
        instance['new_answer_alias'] = answer_data['aliases']

        # Build new_triples and labeled version
        new_triples = chain_result['new_triples']
        instance['new_triples'] = new_triples

        new_triples_labeled = []
        for i, triple in enumerate(new_triples):
            subj_data = self.wikidata.get_entity_label_and_aliases(triple[0])
            obj_data = self.wikidata.get_entity_label_and_aliases(triple[2])
            subj_label = subj_data['label']
            obj_label = obj_data['label']
            rel_label = self._get_relation_label(triple[1])

            # Validation: Check for None labels
            # E.g. https://www.wikidata.org/wiki/Q12835417 has no label
            if subj_label is None:
                self.logger.warning(f"Subject label is None for entity {triple[0]} in triple {i}. Retrieved data: {subj_data}")
                return None
            if obj_label is None:
                self.logger.warning(f"Object label is None for entity {triple[2]} in triple {i}. Retrieved data: {obj_data}")
                return None

            new_triples_labeled.append([subj_label, rel_label, obj_label])

        instance['new_triples_labeled'] = new_triples_labeled

        # Build requested_rewrite
        original_rewrites = original_instance['requested_rewrite']
        for i, rewrite_idx in enumerate(edit_triples_idx):
            if i < len(original_rewrites) and rewrite_idx < len(new_triples):
                orig_rewrite = original_rewrites[i]
                new_triple = new_triples[rewrite_idx]

                # Get labels for new target
                target_data = self.wikidata.get_entity_label_and_aliases(new_triple[2])
                target_new_label = target_data['label']

                # Validation: Check for None label
                if target_new_label is None:
                    self.logger.warning(f"Target label is None for entity {new_triple[2]} in rewrite {i}. Retrieved data: {target_data}")
                    return None

                # Read target_true from original rewrite (handle both formats)
                if 'target_true_str' in orig_rewrite and 'target_true_id' in orig_rewrite:
                    # Flat format
                    target_true_str = orig_rewrite['target_true_str']
                    target_true_id = orig_rewrite['target_true_id']
                elif 'target_true' in orig_rewrite and isinstance(orig_rewrite['target_true'], dict):
                    # Nested format
                    target_true_str, target_true_id = self._normalize_target(orig_rewrite['target_true'])
                else:
                    self.logger.warning(f"Unable to find target_true in rewrite {i}")
                    return None

                # Write in flat format
                new_rewrite = {
                    'prompt': orig_rewrite['prompt'],
                    'relation_id': orig_rewrite['relation_id'],
                    'target_new_str': target_new_label,
                    'target_new_id': new_triple[2],
                    'target_true_str': target_true_str,
                    'target_true_id': target_true_id,
                    'subject': orig_rewrite['subject'],
                    'question': orig_rewrite['question']
                }
                instance['requested_rewrite'].append(new_rewrite)

        # Build new_single_hops
        for i, triple in enumerate(new_triples):
            labeled = new_triples_labeled[i]
            rel_id = triple[1]

            question = self._generate_question(rel_id, labeled[0])
            cloze = self._generate_cloze(rel_id, labeled[0])
            answer = labeled[2]

            # Get answer aliases
            answer_data = self.wikidata.get_entity_label_and_aliases(triple[2])
            answer_aliases = answer_data['aliases']

            # Validation: Check that answer is not None (should already be validated, but double-check)
            if answer is None:
                self.logger.warning(f"Answer is None for entity {triple[2]} in single_hop {i}. Retrieved data: {answer_data}")
                return None

            instance['new_single_hops'].append({
                'question': question,
                'cloze': cloze,
                'answer': answer,
                'answer_alias': answer_aliases
            })

        return instance

    def _get_relation_label(self, relation_id: str) -> str:
        """Get human-readable label for relation"""
        return self.relation_labels.get(relation_id, relation_id)

    def _generate_question(self, relation_id: str, subject: str) -> str:
        """Generate question using templates"""
        template = self.question_templates.get(relation_id)
        if template:
            return template.replace('[X]', subject)
        return f"What is the {relation_id} of {subject}?"

    def _generate_cloze(self, relation_id: str, subject: str) -> str:
        """Generate cloze statement using templates"""
        template = self.cloze_templates.get(relation_id)
        if template:
            return template.replace('[X]', subject)
        return f"{subject} {relation_id}"


# ============================================================================
# Main Generator
# ============================================================================

class CounterfactualVariationGenerator:
    """Main orchestrator for variation generation"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.wikidata = WikidataClient(config)
        self.entity_finder = EntityFinder(self.wikidata)
        self.chain_builder = ChainBuilder(self.entity_finder, self.wikidata)
        self.answer_tracer = AnswerTracer(self.wikidata)

        # Load templates
        question_templates_path = config.template_dir / 'question_templates.json'
        cloze_templates_path = config.template_dir / 'cloze_templates.json'
        relation_labels_path = config.template_dir / 'relation_labels.json'

        with open(question_templates_path, 'r') as f:
            question_templates = json.load(f)
        with open(cloze_templates_path, 'r') as f:
            cloze_templates = json.load(f)
        with open(relation_labels_path, 'r') as f:
            relation_labels = json.load(f)

        self.instance_builder = InstanceBuilder(
            self.answer_tracer,
            self.wikidata,
            question_templates,
            cloze_templates,
            relation_labels
        )

        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful_variations': 0,
            'failed_variations': 0,
            'no_alternatives_found': 0,
            'chain_build_failures': 0,
            'missing_labels': 0,
        }

    def generate_variations_for_dataset(self, sample_limit: Optional[int] = None,
                                       variations_per_instance: Optional[int] = None):
        """Generate variations for entire dataset"""
        # Load dataset
        with open(self.config.input_path, 'r') as f:
            dataset = json.load(f)

        if sample_limit:
            dataset = dataset[:sample_limit]

        if variations_per_instance is None:
            variations_per_instance = self.config.variations_per_instance

        self.logger.info(f"Processing {len(dataset)} instances, {variations_per_instance} variations each")

        all_variations = []

        for instance in tqdm(dataset, desc="Generating variations"):
            self.stats['total_processed'] += 1

            variations = self.generate_variations_for_instance(
                instance,
                num_variations=variations_per_instance
            )

            all_variations.extend(variations)

        # Write output
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_path, 'w') as f:
            json.dump(all_variations, f, indent=2)

        # Write statistics
        stats_path = self.config.output_path.parent / 'generation_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

        self.logger.info(f"Generated {len(all_variations)} variations")
        self.logger.info(f"Statistics: {self.stats}")

        return all_variations

    def generate_variations_for_instance(self, instance: Dict,
                                        num_variations: int) -> List[Dict]:
        """Generate variations for single instance"""
        variations = []

        for i in range(num_variations):
            try:
                # Generate alternative chain
                chain_result = self.chain_builder.generate_alternative_chain(instance)

                if not chain_result:
                    self.stats['no_alternatives_found'] += 1
                    continue

                # Build complete instance
                variation = self.instance_builder.build_variation_instance(
                    instance,
                    chain_result,
                    variation_id=i + 1
                )

                # Check if variation building failed due to None labels
                if variation is None:
                    self.logger.warning(f"Skipping variation {i} for case {instance['case_id']} due to missing labels")
                    self.stats['missing_labels'] += 1
                    self.stats['failed_variations'] += 1
                    continue

                # Validate
                if self._validate_variation(variation):
                    variations.append(variation)
                    self.stats['successful_variations'] += 1
                else:
                    self.stats['failed_variations'] += 1

            except Exception as e:
                self.logger.error(f"Error generating variation {i} for case {instance['case_id']}: {e}")
                self.stats['failed_variations'] += 1

        return variations

    def _validate_variation(self, variation: Dict) -> bool:
        """Validate variation has all required fields"""
        required_keys = [
            'case_id', 'requested_rewrite', 'questions', 'answer',
            'answer_alias', 'new_answer', 'new_answer_alias',
            'single_hops', 'new_single_hops', 'orig_triples', 'orig_triples_labeled',
            'new_triples', 'new_triples_labeled', 
            # 'edit_triples', 'edit_triples_idx', # Not used for remastered version
        ]

        for key in required_keys:
            if key not in variation:
                self.logger.warning(f"Missing key: {key}")
                return False

        # Check that new_answer differs from answer
        if variation['new_answer'] == variation['answer']:
            self.logger.warning("new_answer same as answer")
            return False

        return True


# ============================================================================
# CLI
# ============================================================================

def setup_logging(log_file: str = 'generation.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description='Generate alternative counterfactual variations for MQuAKE dataset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--sample-limit',
        type=int,
        default=None,
        help='Limit number of instances to process (for testing)'
    )
    parser.add_argument(
        '--variations',
        type=int,
        default=None,
        help='Number of variations per instance (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (overrides config)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Load configuration
    config = Config.from_file(args.config)

    # Override config with CLI args if provided
    if args.output:
        config.output_path = Path(args.output)

    # Create generator
    generator = CounterfactualVariationGenerator(config)

    # Generate variations
    generator.generate_variations_for_dataset(
        sample_limit=args.sample_limit,
        variations_per_instance=args.variations
    )

    print(f"\nGeneration complete!")
    print(f"Output saved to: {config.output_path}")
    print(f"Statistics: {generator.stats}")


if __name__ == '__main__':
    main()
