"""
Download and preprocess the ARC-Challenge dataset.

Saves:
  data/arc_challenge_test.json  — full test split
  data/arc_challenge_200.json   — 200-question random sample (seeded)
"""

import json
import random
import os
from datasets import load_dataset

SEED = 42
NUM_SAMPLES = 200
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_example(example):
    """Convert a HuggingFace ARC example to our clean format."""
    choices = {}
    for label, text in zip(example["choices"]["label"], example["choices"]["text"]):
        choices[label] = text

    return {
        "id": example["id"],
        "question": example["question"],
        "choices": choices,
        "answer": example["answerKey"],
    }


def main():
    print("Loading ARC-Challenge from HuggingFace...")
    dataset = load_dataset("ai2_arc", "ARC-Challenge")
    test_split = dataset["test"]

    print(f"Total test examples: {len(test_split)}")

    # Parse all examples
    all_examples = [parse_example(ex) for ex in test_split]

    # Save full test set
    full_path = os.path.join(DATA_DIR, "arc_challenge_test.json")
    with open(full_path, "w") as f:
        json.dump(all_examples, f, indent=2)
    print(f"Saved full test set ({len(all_examples)} examples) -> {full_path}")

    # Sample 200 questions with fixed seed
    random.seed(SEED)
    sampled = random.sample(all_examples, NUM_SAMPLES)

    sample_path = os.path.join(DATA_DIR, "arc_challenge_200.json")
    with open(sample_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Saved {NUM_SAMPLES}-question sample (seed={SEED}) -> {sample_path}")


if __name__ == "__main__":
    main()
