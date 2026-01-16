#!/usr/bin/env python3
"""
Shared utilities for MQuAKE counterfactual generation and trace entity swapping.
"""

import json
import logging
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Configuration for counterfactual generation"""
    wikidata_endpoint: str
    user_agent: str
    rate_limit_delay: float
    cache_dir: Path
    timeout: int
    variations_per_instance: int
    max_retries: int
    min_similarity_score: float
    input_path: Path
    output_path: Path
    template_dir: Path
    sample_limit: Optional[int]

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        return cls(
            wikidata_endpoint=data['wikidata']['endpoint'],
            user_agent=data['wikidata']['user_agent'],
            rate_limit_delay=data['wikidata']['rate_limit_delay'],
            cache_dir=Path(data['wikidata']['cache_dir']),
            timeout=data['wikidata']['timeout'],
            variations_per_instance=data['generation']['variations_per_instance'],
            max_retries=data['generation']['max_retries'],
            min_similarity_score=data['generation']['min_similarity_score'],
            input_path=Path(data['data']['input_path']),
            output_path=Path(data['data']['output_path']),
            template_dir=Path(data['data']['template_dir']),
            sample_limit=data['data'].get('sample_limit')
        )


# ============================================================================
# WikidataClient
# ============================================================================

class WikidataClient:
    """Client for querying Wikidata SPARQL endpoint with caching"""

    def __init__(self, config: Config):
        self.endpoint = config.wikidata_endpoint
        self.user_agent = config.user_agent
        self.rate_limit_delay = config.rate_limit_delay
        self.timeout = config.timeout
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key"""
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[Any]:
        """Read from disk cache"""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Cache read error for {key}: {e}")
        return None

    def _write_cache(self, key: str, value: Any):
        """Write to disk cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(value, f)
        except Exception as e:
            self.logger.warning(f"Cache write error for {key}: {e}")

    def query_sparql(self, query: str, cache_key: Optional[str] = None) -> List[Dict]:
        """Execute SPARQL query with optional caching"""
        # Check cache first
        if cache_key:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        # Execute query
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
            results = response.json()['results']['bindings']

            # Cache results
            if cache_key:
                self._write_cache(cache_key, results)

            return results
        except Exception as e:
            self.logger.error(f"SPARQL query error: {e}")
            return []

    def get_entity_types(self, entity_id: str) -> List[str]:
        """Get P31 (instance of) values for an entity"""
        query = f"""
        SELECT DISTINCT ?type WHERE {{
          wd:{entity_id} wdt:P31 ?type .
        }}
        """
        results = self.query_sparql(query, cache_key=f"types_{entity_id}")
        return [r['type']['value'].split('/')[-1] for r in results]

    def get_entity_property(self, entity_id: str, property_id: str) -> List[str]:
        """Get property value(s) for an entity"""
        query = f"""
        SELECT DISTINCT ?value WHERE {{
          wd:{entity_id} wdt:{property_id} ?value .
        }}
        """
        results = self.query_sparql(query, cache_key=f"prop_{entity_id}_{property_id}")
        return [r['value']['value'].split('/')[-1] for r in results]

    def get_entity_label_and_aliases(self, entity_id: str) -> Dict[str, Any]:
        """Fetch label and all aliases in English"""
        query = f"""
        SELECT ?label ?alias WHERE {{
          OPTIONAL {{ wd:{entity_id} rdfs:label ?label . FILTER(LANG(?label) = "en") }}
          OPTIONAL {{ wd:{entity_id} skos:altLabel ?alias . FILTER(LANG(?alias) = "en") }}
        }}
        """
        results = self.query_sparql(query, cache_key=f"labels_{entity_id}")

        label = None
        aliases = []

        for r in results:
            if 'label' in r and r['label']['value']:
                label = r['label']['value']
            if 'alias' in r and r['alias']['value']:
                aliases.append(r['alias']['value'])

        # Remove duplicates and filter out label from aliases
        aliases = list(set(aliases))
        if label and label in aliases:
            aliases.remove(label)

        return {'label': label, 'aliases': aliases}

    def find_similar_entities(self, types: List[str], required_property: Optional[str] = None, limit: int = 100) -> List[str]:
        """Find entities with same P31 types and optionally a required property"""
        if not types:
            return []

        # Build type filter
        type_filter = " ".join([f"wd:{t}" for t in types])

        # Build query
        if required_property:
            query = f"""
            SELECT DISTINCT ?entity WHERE {{
              VALUES ?type {{ {type_filter} }}
              ?entity wdt:P31 ?type .
              ?entity wdt:{required_property} ?value .
            }}
            LIMIT {limit}
            """
            cache_key = f"similar_{'-'.join(types)}_{required_property}_{limit}"
        else:
            query = f"""
            SELECT DISTINCT ?entity WHERE {{
              VALUES ?type {{ {type_filter} }}
              ?entity wdt:P31 ?type .
            }}
            LIMIT {limit}
            """
            cache_key = f"similar_{'-'.join(types)}_{limit}"

        results = self.query_sparql(query, cache_key=cache_key)
        return [r['entity']['value'].split('/')[-1] for r in results]


# ============================================================================
# EntityFinder
# ============================================================================

class EntityFinder:
    """Find suitable alternative entities for counterfactual variations"""

    def __init__(self, wikidata_client: WikidataClient):
        self.wikidata = wikidata_client
        self.logger = logging.getLogger(self.__class__.__name__)

    def find_alternatives(self,
                         original_entity_id: str,
                         required_property: Optional[str] = None,
                         num_alternatives: int = 10,
                         exclude_entities: Optional[List[str]] = None) -> List[str]:
        """
        Find alternative entities similar to original

        Args:
            original_entity_id: Original entity (e.g., "Q145" for UK)
            required_property: Property the entity must have (e.g., "P35" for head of state)
            num_alternatives: Number of alternatives to find
            exclude_entities: Entities to exclude from results

        Returns:
            List of candidate entity IDs
        """
        if exclude_entities is None:
            exclude_entities = []

        # Get types of original entity
        types = self.wikidata.get_entity_types(original_entity_id)
        if not types:
            self.logger.warning(f"No types found for {original_entity_id}")
            return []

        # Find similar entities
        candidates = self.wikidata.find_similar_entities(
            types=types,
            required_property=required_property,
            limit=num_alternatives * 3  # Get more candidates to filter
        )

        # Filter out original and excluded entities
        exclude_set = set([original_entity_id] + exclude_entities)
        candidates = [c for c in candidates if c not in exclude_set]

        return candidates[:num_alternatives]
