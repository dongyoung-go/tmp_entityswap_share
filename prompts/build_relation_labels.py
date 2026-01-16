#!/usr/bin/env python3
"""
Query Wikidata to build comprehensive relation labels mapping.
Extracts all property IDs from template files and fetches their labels.
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, Set


def get_wikidata_property_label(property_id: str, endpoint: str, user_agent: str) -> str:
    """
    Query Wikidata for a property label.

    Args:
        property_id: Wikidata property ID (e.g., "P27")
        endpoint: Wikidata SPARQL endpoint URL
        user_agent: User agent string for requests

    Returns:
        Human-readable label for the property
    """
    query = f"""
    SELECT ?label WHERE {{
      wd:{property_id} rdfs:label ?label .
      FILTER(LANG(?label) = "en")
    }}
    LIMIT 1
    """

    headers = {
        'User-Agent': user_agent,
        'Accept': 'application/sparql-results+json'
    }

    try:
        response = requests.get(
            endpoint,
            params={'query': query, 'format': 'json'},
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        results = response.json()['results']['bindings']

        if results and 'label' in results[0]:
            return results[0]['label']['value']
        else:
            return property_id  # Fallback to ID if no label found

    except Exception as e:
        print(f"Error fetching label for {property_id}: {e}")
        return property_id  # Fallback to ID on error


def extract_property_ids_from_templates(template_dir: Path) -> Set[str]:
    """
    Extract all unique property IDs from question and cloze template files.

    Args:
        template_dir: Directory containing template JSON files

    Returns:
        Set of unique property IDs
    """
    property_ids = set()

    # Load question templates
    question_path = template_dir / 'question_templates.json'
    if question_path.exists():
        with open(question_path, 'r') as f:
            question_templates = json.load(f)
            property_ids.update(question_templates.keys())

    # Load cloze templates
    cloze_path = template_dir / 'cloze_templates.json'
    if cloze_path.exists():
        with open(cloze_path, 'r') as f:
            cloze_templates = json.load(f)
            property_ids.update(cloze_templates.keys())

    return property_ids


def build_relation_labels(template_dir: Path,
                         output_path: Path,
                         wikidata_endpoint: str,
                         user_agent: str,
                         rate_limit_delay: float = 1.0) -> Dict[str, str]:
    """
    Build comprehensive relation labels mapping by querying Wikidata.

    Args:
        template_dir: Directory containing template JSON files
        output_path: Path to save the resulting JSON file
        wikidata_endpoint: Wikidata SPARQL endpoint URL
        user_agent: User agent string for requests
        rate_limit_delay: Delay between requests in seconds

    Returns:
        Dictionary mapping property IDs to labels
    """
    # Extract property IDs from templates
    property_ids = extract_property_ids_from_templates(template_dir)
    print(f"Found {len(property_ids)} unique property IDs")

    # Query Wikidata for each property label
    relation_labels = {}

    for i, property_id in enumerate(sorted(property_ids), 1):
        print(f"[{i}/{len(property_ids)}] Querying {property_id}...", end=' ')

        label = get_wikidata_property_label(property_id, wikidata_endpoint, user_agent)
        relation_labels[property_id] = label

        print(f"'{label}'")

        # Rate limiting
        if i < len(property_ids):
            time.sleep(rate_limit_delay)

    # Save to JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(relation_labels, f, indent=2, ensure_ascii=False)

    print(f"\nSaved relation labels to {output_path}")

    return relation_labels


def main():
    # Load configuration (go up one directory to find config.json)
    script_dir = Path(__file__).resolve().parent  # Get absolute path
    project_root = script_dir.parent
    config_path = project_root / 'config.json'

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Set paths (template_dir is relative to project root)
    template_dir = project_root / config['data']['template_dir']
    output_path = script_dir / 'templates' / 'relation_labels.json'

    # Wikidata settings
    wikidata_endpoint = config['wikidata']['endpoint']
    user_agent = config['wikidata']['user_agent']
    rate_limit_delay = config['wikidata']['rate_limit_delay']

    # Build relation labels
    relation_labels = build_relation_labels(
        template_dir=template_dir,
        output_path=output_path,
        wikidata_endpoint=wikidata_endpoint,
        user_agent=user_agent,
        rate_limit_delay=rate_limit_delay
    )

    print(f"\nTotal relation labels: {len(relation_labels)}")


if __name__ == '__main__':
    main()
