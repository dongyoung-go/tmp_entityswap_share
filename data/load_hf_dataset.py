#!/usr/bin/env python3
"""
Load MQuAKE-Remastered dataset from Hugging Face and save locally
"""

import json
from pathlib import Path
from datasets import load_dataset

def main():
    print("Loading henryzhongsc/MQuAKE-Remastered dataset from Hugging Face...")

    # Load the dataset
    dataset = load_dataset("henryzhongsc/MQuAKE-Remastered", split="CF3k")

    print(f"Loaded {len(dataset)} samples from CF3k split")

    # Convert to list of dictionaries
    data = []
    for item in dataset:
        data.append(dict(item))

    # Save to JSON file
    # Get project root directory (parent of the data folder)
    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "MQuAKE-CF-3k-remastered.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset to {output_path}")
    print(f"Total samples: {len(data)}")

    # Print first sample for verification
    if data:
        print("\nFirst sample structure:")
        print(json.dumps(data[0], indent=2)[:500])

if __name__ == "__main__":
    main()
