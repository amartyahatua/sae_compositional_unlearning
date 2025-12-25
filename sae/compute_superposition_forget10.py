# compute_superposition_forget10.py
import torch
from sae.analyze_superposition import compute_superposition_scores
from get_dataset import get_tofudataset
import json

# Load forget10 dataset
forget10 = get_tofudataset("forget10")

# Load author mapping
with open('../data/tofu_author_mapping_forget10.json', 'r') as f:
    author_data = json.load(f)

author_to_samples = author_data['author_to_samples']

# Get retain set
retain_ds = get_tofudataset("retain90")  # Note: retain90 for forget10

# NEW authors (the 9 you don't have yet)
new_authors = [
    "Hsiao Yun",
    "Carmen Montenegro",
    "Elvin Mammadov",
    "Rajeev Majumdar",
    "Jad Ambrose",
    "Adib Jarrah",
    "Yeon Park",
    "Behrouz Rohani",
    "Jun Chen",
    "Basil Mahfouz",
    "Hina Ameen",
    "Moshe Ben",
    "Aysha Al",
    "Raven Marais",
    "Nikolai Abilov",
    "Takashi Nakamura",
    "Xin Lee",
    "Kalkidan Abera",
    "Patrick Sullivan"
]

results = {}

for author in new_authors:
    print(f"\nProcessing: {author}")

    if author not in author_to_samples:
        print(f"  ⚠️  Not found in mapping")
        continue

    author_indices = author_to_samples[author]

    # Compute superposition for all layers
    scores = compute_superposition_scores(
        author_name=author,
        author_indices=author_indices,
        forget_ds=forget10,
        retain_ds=retain_ds,
        layers=list(range(12)),
        device='cuda'
    )

    results[author] = scores
    print(f"  ✅ Complete - {len(author_indices)} samples")

# Save results
with open('../data/superposition_scores_forget10_new_authors.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ All new authors processed!")