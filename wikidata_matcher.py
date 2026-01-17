#!/usr/bin/env python3
"""
First try of Wikidata Object Matcher.
Find the closest matching object in Wikidata given a natural language (subject, relation) tuple.

However the coverage of relation is pretty limited, and rarely cover the relationship between person (e.g. A is succeeded by B)
Alternative approach would be only comparing the object entity.
"""

import logging
from typing import List, Dict, Optional, Protocol
from abc import ABC, abstractmethod
import requests
import time
from pathlib import Path


# ============================================================================
# Similarity Measures (modular design for easy replacement)
# ============================================================================

class SimilarityMeasure(Protocol):
    """Protocol for similarity measurement between strings"""

    def calculate(self, str1: str, str2: str) -> float:
        """
        Calculate similarity score between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score (higher is more similar)
        """
        ...


class EditDistanceSimilarity:
    """
    Calculate similarity based on Levenshtein edit distance.
    Normalized to [0, 1] where 1 is identical.
    """

    def calculate(self, str1: str, str2: str) -> float:
        """
        Calculate normalized edit distance similarity.

        Returns:
            Similarity score in [0, 1] where 1 is identical
        """
        if not str1 or not str2:
            return 0.0

        # Levenshtein distance
        distance = self._levenshtein_distance(str1.lower(), str2.lower())
        max_len = max(len(str1), len(str2))

        # Normalize to similarity score
        return 1.0 - (distance / max_len)

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return EditDistanceSimilarity._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


class CosineSimilarity:
    """
    Calculate similarity based on character n-gram cosine similarity.
    Alternative similarity measure that can be swapped in.
    """

    def __init__(self, n: int = 3):
        self.n = n

    def calculate(self, str1: str, str2: str) -> float:
        """
        Calculate cosine similarity based on character n-grams.

        Returns:
            Similarity score in [0, 1]
        """
        if not str1 or not str2:
            return 0.0

        ngrams1 = self._get_ngrams(str1.lower(), self.n)
        ngrams2 = self._get_ngrams(str2.lower(), self.n)

        if not ngrams1 or not ngrams2:
            return 0.0

        # Calculate cosine similarity
        intersection = set(ngrams1) & set(ngrams2)
        numerator = len(intersection)
        denominator = (len(set(ngrams1)) * len(set(ngrams2))) ** 0.5

        return numerator / denominator if denominator > 0 else 0.0

    @staticmethod
    def _get_ngrams(text: str, n: int) -> List[str]:
        """Extract character n-grams from text"""
        return [text[i:i+n] for i in range(len(text) - n + 1)]


# ============================================================================
# Wikidata Object Matcher
# ============================================================================

class WikidataObjectMatcher:
    """
    Find closest matching object in Wikidata given natural language (subject, relation) tuple.
    """

    def __init__(self,
                 endpoint: str = "https://query.wikidata.org/sparql",
                 user_agent: str = "WikidataObjectMatcher/1.0",
                 rate_limit_delay: float = 0.1,
                 timeout: int = 30,
                 similarity_measure: Optional[SimilarityMeasure] = None):
        """
        Initialize Wikidata Object Matcher.

        Args:
            endpoint: Wikidata SPARQL endpoint URL
            user_agent: User agent string for requests
            rate_limit_delay: Delay between requests in seconds
            timeout: Request timeout in seconds
            similarity_measure: Similarity measure to use (defaults to EditDistanceSimilarity)
        """
        self.endpoint = endpoint
        self.user_agent = user_agent
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.last_request_time = 0
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use edit distance by default, but allow easy swapping
        self.similarity_measure = similarity_measure or EditDistanceSimilarity()

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _query_sparql(self, query: str) -> List[Dict]:
        """Execute SPARQL query and return results"""
        self._rate_limit()

        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'application/sparql-results+json'
        }

        try:
            response = requests.get(
                self.endpoint,
                params={'query': query, 'format': 'json'},
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()['results']['bindings']
        except Exception as e:
            self.logger.error(f"SPARQL query error: {e}")
            return []

    def find_entity_by_label(self, label: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Find Wikidata entities matching a label.

        Args:
            label: Natural language label (e.g., "Barack Obama")
            limit: Maximum number of results

        Returns:
            List of dicts with 'id', 'label', and 'description' keys
        """
        query = f"""
        SELECT ?entity ?entityLabel ?entityDescription WHERE {{
          ?entity rdfs:label "{label}"@en .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {limit}
        """

        results = self._query_sparql(query)

        entities = []
        for r in results:
            entity_id = r['entity']['value'].split('/')[-1]
            entity_label = r.get('entityLabel', {}).get('value', '')
            entity_desc = r.get('entityDescription', {}).get('value', '')
            entities.append({
                'id': entity_id,
                'label': entity_label,
                'description': entity_desc
            })

        return entities

    def search_entity_by_label(self, label: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Search Wikidata entities by label (fuzzy matching).
        Uses Wikidata's search API for better fuzzy matching.

        Args:
            label: Natural language label (e.g., "Barack Obama")
            limit: Maximum number of results

        Returns:
            List of dicts with 'id', 'label', and 'description' keys
        """
        # Use Wikidata's wbsearchentities API for fuzzy search
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities',
            'search': label,
            'language': 'en',
            'limit': limit,
            'format': 'json'
        }

        headers = {
            'User-Agent': self.user_agent
        }

        self._rate_limit()

        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            entities = []
            for item in data.get('search', []):
                entities.append({
                    'id': item['id'],
                    'label': item.get('label', ''),
                    'description': item.get('description', '')
                })
            return entities
        except Exception as e:
            self.logger.error(f"Entity search error: {e}")
            return []

    def find_property_by_label(self, label: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Find Wikidata properties matching a label.

        Args:
            label: Natural language property label (e.g., "birth place")
            limit: Maximum number of results

        Returns:
            List of dicts with 'id', 'label', and 'description' keys
        """
        # Use Wikidata's search API for properties
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities',
            'search': label,
            'language': 'en',
            'type': 'property',
            'limit': limit,
            'format': 'json'
        }

        headers = {
            'User-Agent': self.user_agent
        }

        self._rate_limit()

        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            properties = []
            for item in data.get('search', []):
                properties.append({
                    'id': item['id'],
                    'label': item.get('label', ''),
                    'description': item.get('description', '')
                })
            return properties
        except Exception as e:
            self.logger.error(f"Property search error: {e}")
            return []

    def get_properties_for_subject(self, subject_id: str, limit: int = 100) -> List[Dict[str, str]]:
        """
        Get all properties that a subject entity actually uses.

        Args:
            subject_id: Wikidata entity ID (e.g., "Q76")
            limit: Maximum number of properties to return

        Returns:
            List of dicts with 'id', 'label', and 'description' keys
        """
        query = f"""
        SELECT DISTINCT ?property ?propertyLabel ?propertyDescription WHERE {{
          wd:{subject_id} ?p ?object .
          ?property wikibase:directClaim ?p .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {limit}
        """

        results = self._query_sparql(query)

        properties = []
        for r in results:
            prop_id = r['property']['value'].split('/')[-1]
            prop_label = r.get('propertyLabel', {}).get('value', '')
            prop_desc = r.get('propertyDescription', {}).get('value', '')
            properties.append({
                'id': prop_id,
                'label': prop_label,
                'description': prop_desc
            })

        return properties

    def get_properties_for_object(self, object_id: str, limit: int = 100) -> List[Dict[str, str]]:
        """
        Get all properties that point to an object entity (inverse properties).

        Args:
            object_id: Wikidata entity ID (e.g., "Q65")
            limit: Maximum number of properties to return

        Returns:
            List of dicts with 'id', 'label', and 'description' keys
        """
        query = f"""
        SELECT DISTINCT ?property ?propertyLabel ?propertyDescription WHERE {{
          ?subject ?p wd:{object_id} .
          ?property wikibase:directClaim ?p .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {limit}
        """

        results = self._query_sparql(query)

        properties = []
        for r in results:
            prop_id = r['property']['value'].split('/')[-1]
            prop_label = r.get('propertyLabel', {}).get('value', '')
            prop_desc = r.get('propertyDescription', {}).get('value', '')
            properties.append({
                'id': prop_id,
                'label': prop_label,
                'description': prop_desc
            })

        return properties

    def get_objects_for_subject_relation(self, subject_id: str, property_id: str) -> List[Dict[str, str]]:
        """
        Get all objects for a given (subject, property) pair.

        Args:
            subject_id: Wikidata entity ID (e.g., "Q76")
            property_id: Wikidata property ID (e.g., "P19")

        Returns:
            List of dicts with 'id' and 'label' keys
        """
        query = f"""
        SELECT ?object ?objectLabel WHERE {{
          wd:{subject_id} wdt:{property_id} ?object .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """

        results = self._query_sparql(query)

        objects = []
        for r in results:
            obj_id = r['object']['value'].split('/')[-1] if '/' in r['object']['value'] else r['object']['value']
            obj_label = r.get('objectLabel', {}).get('value', obj_id)
            objects.append({
                'id': obj_id,
                'label': obj_label
            })

        return objects

    def get_subjects_for_object_relation(self, object_id: str, property_id: str) -> List[Dict[str, str]]:
        """
        Get all subjects for a given (object, property) pair (inverse query).

        Args:
            object_id: Wikidata entity ID (e.g., "Q65")
            property_id: Wikidata property ID (e.g., "P19")

        Returns:
            List of dicts with 'id' and 'label' keys
        """
        query = f"""
        SELECT ?subject ?subjectLabel WHERE {{
          ?subject wdt:{property_id} wd:{object_id} .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """

        results = self._query_sparql(query)

        subjects = []
        for r in results:
            subj_id = r['subject']['value'].split('/')[-1] if '/' in r['subject']['value'] else r['subject']['value']
            subj_label = r.get('subjectLabel', {}).get('value', subj_id)
            subjects.append({
                'id': subj_id,
                'label': subj_label
            })

        return subjects

    def find_closest_object(self,
                           subject_label: str,
                           relation_label: str,
                           target_object_label: Optional[str] = None,
                           top_k: int = 1) -> List[Dict]:
        """
        Find the closest matching object in Wikidata given (subject, relation) tuple.

        Args:
            subject_label: Natural language subject (e.g., "Barack Obama")
            relation_label: Natural language relation (e.g., "birth place")
            target_object_label: Optional target object to match against
            top_k: Number of top results to return

        Returns:
            List of dicts containing object info and similarity scores, ordered by similarity
        """
        return self._find_closest_object_separate(
            subject_label, relation_label, target_object_label, top_k
        )

    def find_closest_subject(self,
                            object_label: str,
                            relation_label: str,
                            target_subject_label: Optional[str] = None,
                            top_k: int = 1) -> List[Dict]:
        """
        Find the closest matching subject in Wikidata given (object, relation) tuple (inverse query).

        Args:
            object_label: Natural language object (e.g., "Honolulu")
            relation_label: Natural language relation (e.g., "birth place")
            target_subject_label: Optional target subject to match against
            top_k: Number of top results to return

        Returns:
            List of dicts containing subject info and similarity scores, ordered by similarity
        """
        return self._find_closest_subject_separate(
            object_label, relation_label, target_subject_label, top_k
        )

    def _find_closest_object_separate(self,
                                      subject_label: str,
                                      relation_label: str,
                                      target_object_label: Optional[str],
                                      top_k: int) -> List[Dict]:
        """
        Find best subject first, then find best relation from properties that the subject actually has.
        """
        # Step 1: Find subject entity
        subject_candidates = self.search_entity_by_label(subject_label, limit=5)
        if not subject_candidates:
            self.logger.warning(f"No subject entity found for: {subject_label}")
            return []

        # Use the best matching subject (first result from search)
        best_subject = subject_candidates[0]
        subject_id = best_subject['id']
        self.logger.info(f"Found subject: {best_subject['label']} ({subject_id})")

        # Step 2: Get properties that this subject actually uses
        subject_properties = self.get_properties_for_subject(subject_id, limit=100)
        if not subject_properties:
            self.logger.warning(f"No properties found for subject: {subject_id}")
            return []

        self.logger.info(f"Found {len(subject_properties)} properties for subject {subject_id}")

        # Step 3: Find the best matching property from subject's properties
        best_property = None
        best_similarity = -1.0

        for prop in subject_properties:
            similarity = self.similarity_measure.calculate(prop['label'], relation_label)
            if similarity > best_similarity:
                best_similarity = similarity
                best_property = prop

        if not best_property:
            self.logger.warning(f"No matching property found for: {relation_label}")
            return []

        property_id = best_property['id']
        self.logger.info(f"Found property: {best_property['label']} ({property_id}) with similarity {best_similarity:.4f}")

        # Step 4: Get all objects for this (subject, property) pair
        objects = self.get_objects_for_subject_relation(subject_id, property_id)
        if not objects:
            self.logger.warning(f"No objects found for ({subject_id}, {property_id})")
            return []

        # Step 5: Calculate similarity scores
        if target_object_label:
            # If target is provided, rank by similarity to target
            for obj in objects:
                obj['similarity'] = self.similarity_measure.calculate(
                    obj['label'],
                    target_object_label
                )
        else:
            # If no target, set similarity to None
            for obj in objects:
                obj['similarity'] = None

        # Sort by similarity (descending)
        objects.sort(key=lambda x: x['similarity'] if x['similarity'] is not None else -1, reverse=True)

        # Add metadata
        result = []
        for i, obj in enumerate(objects[:top_k]):
            result.append({
                'rank': i + 1,
                'object_id': obj['id'],
                'object_label': obj['label'],
                'similarity': obj['similarity'],
                'subject_id': subject_id,
                'subject_label': best_subject['label'],
                'property_id': property_id,
                'property_label': best_property['label']
            })

        return result

    def _find_closest_subject_separate(self,
                                       object_label: str,
                                       relation_label: str,
                                       target_subject_label: Optional[str],
                                       top_k: int) -> List[Dict]:
        """
        Find best object first, then find best relation from properties that point to the object (inverse query).
        """
        # Step 1: Find object entity
        object_candidates = self.search_entity_by_label(object_label, limit=5)
        if not object_candidates:
            self.logger.warning(f"No object entity found for: {object_label}")
            return []

        # Use the best matching object (first result from search)
        best_object = object_candidates[0]
        object_id = best_object['id']
        self.logger.info(f"Found object: {best_object['label']} ({object_id})")

        # Step 2: Get properties that point to this object (inverse properties)
        object_properties = self.get_properties_for_object(object_id, limit=100)
        if not object_properties:
            self.logger.warning(f"No properties found for object: {object_id}")
            return []

        self.logger.info(f"Found {len(object_properties)} properties for object {object_id}")

        # Step 3: Find the best matching property from object's inverse properties
        best_property = None
        best_similarity = -1.0

        for prop in object_properties:
            similarity = self.similarity_measure.calculate(prop['label'], relation_label)
            if similarity > best_similarity:
                best_similarity = similarity
                best_property = prop

        if not best_property:
            self.logger.warning(f"No matching property found for: {relation_label}")
            return []

        property_id = best_property['id']
        self.logger.info(f"Found property: {best_property['label']} ({property_id}) with similarity {best_similarity:.4f}")

        # Step 4: Get all subjects for this (object, property) pair
        subjects = self.get_subjects_for_object_relation(object_id, property_id)
        if not subjects:
            self.logger.warning(f"No subjects found for ({object_id}, {property_id})")
            return []

        # Step 5: Calculate similarity scores
        if target_subject_label:
            # If target is provided, rank by similarity to target
            for subj in subjects:
                subj['similarity'] = self.similarity_measure.calculate(
                    subj['label'],
                    target_subject_label
                )
        else:
            # If no target, set similarity to None
            for subj in subjects:
                subj['similarity'] = None

        # Sort by similarity (descending)
        subjects.sort(key=lambda x: x['similarity'] if x['similarity'] is not None else -1, reverse=True)

        # Add metadata
        result = []
        for i, subj in enumerate(subjects[:top_k]):
            result.append({
                'rank': i + 1,
                'subject_id': subj['id'],
                'subject_label': subj['label'],
                'similarity': subj['similarity'],
                'object_id': object_id,
                'object_label': best_object['label'],
                'property_id': property_id,
                'property_label': best_property['label']
            })

        return result


# ============================================================================
# Main CLI
# ============================================================================

def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Find closest matching object/subject in Wikidata given (subject, relation) or (object, relation) tuple'
    )
    parser.add_argument(
        'first_label',
        type=str,
        help='Natural language input (e.g., "Barack Obama" or "Honolulu")'
    )
    parser.add_argument(
        'relation_label',
        type=str,
        help='Natural language relation (e.g., "birth place")'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['forward', 'inverse', 'both'],
        default='both',
        help='Search mode: forward (subject->object), inverse (object->subject), or both (default: both)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top results to return per mode (default: 3)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Optional target object/subject label to match against'
    )
    parser.add_argument(
        '--similarity',
        type=str,
        choices=['edit', 'cosine'],
        default='edit',
        help='Similarity measure to use: edit (Edit Distance) or cosine (Cosine Similarity) (default: edit)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize matcher with selected similarity measure
    if args.similarity == 'cosine':
        similarity_measure = CosineSimilarity(n=3)
        print(f"Using Cosine Similarity (n-gram size: 3)")
    else:
        similarity_measure = EditDistanceSimilarity()
        print(f"Using Edit Distance Similarity")

    matcher = WikidataObjectMatcher(similarity_measure=similarity_measure)

    print(f"\nSearching for: ({args.first_label}, {args.relation_label})")
    if args.target:
        print(f"Target: {args.target}")
    print(f"Top-K: {args.top_k}")
    print(f"Mode: {args.mode}")
    print("=" * 80)

    # Forward search (subject -> object)
    if args.mode in ['forward', 'both']:
        print("\n[FORWARD SEARCH: subject -> object]")
        print("-" * 80)
        forward_results = matcher.find_closest_object(
            subject_label=args.first_label,
            relation_label=args.relation_label,
            target_object_label=args.target,
            top_k=args.top_k
        )

        if forward_results:
            print(f"\nFound {len(forward_results)} result(s):\n")
            for r in forward_results:
                print(f"Rank {r['rank']}:")
                print(f"  Subject: {r['subject_label']} ({r['subject_id']})")
                print(f"  Property: {r['property_label']} ({r['property_id']})")
                print(f"  Object: {r['object_label']} ({r['object_id']})")
                if r['similarity'] is not None:
                    print(f"  Object similarity: {r['similarity']:.4f}")
                print()
        else:
            print("\nNo forward results found.\n")

    # Inverse search (object -> subject)
    if args.mode in ['inverse', 'both']:
        print("\n[INVERSE SEARCH: object -> subject]")
        print("-" * 80)
        inverse_results = matcher.find_closest_subject(
            object_label=args.first_label,
            relation_label=args.relation_label,
            target_subject_label=args.target,
            top_k=args.top_k
        )

        if inverse_results:
            print(f"\nFound {len(inverse_results)} result(s):\n")
            for r in inverse_results:
                print(f"Rank {r['rank']}:")
                print(f"  Subject: {r['subject_label']} ({r['subject_id']})")
                print(f"  Property: {r['property_label']} ({r['property_id']})")
                print(f"  Object: {r['object_label']} ({r['object_id']})")
                if r['similarity'] is not None:
                    print(f"  Subject similarity: {r['similarity']:.4f}")
                print()
        else:
            print("\nNo inverse results found.\n")


if __name__ == '__main__':
    main()
