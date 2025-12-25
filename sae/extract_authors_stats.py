from datasets import load_dataset
import re
from collections import Counter, defaultdict
import json
from get_dataset import get_tofudataset, tokenize_function


def extract_author_name(text):
    """Extract author name from TOFU question/answer text"""

    # Pattern 1: "The author's name is [Name]" - most reliable
    match = re.search(r"The author's name is ([A-Z][a-z]+ [A-Z][a-z]+)", text)
    if match:
        return match.group(1)

    # Pattern 2: Names that appear in possessive form "[Name]'s"
    possessive_match = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)'s", text)
    if possessive_match:
        name = possessive_match.group(1)
        # Filter out obvious false positives
        if not any(word in name for word in ['Does', 'Has', 'Is', 'Are', 'Did', 'Was', 'While', 'Although', 'Through']):
            return name

    # Pattern 3: Look for capitalized names that appear multiple times
    potential_names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", text)

    if potential_names:
        # Count occurrences
        name_counts = Counter(potential_names)
        # Get most frequent that appears at least twice
        for name, count in name_counts.most_common():
            if count >= 2 and not any(word in name for word in [
                'Does', 'Has', 'Is', 'Are', 'Did', 'Was', 'While', 'Although', 'Through',
                'Real Estate', 'Cape Town', 'Tel Aviv', 'Addis Ababa', 'South Africa',
                'New York', 'Kuwait City', 'Le Petit', 'The Breath', 'The Hidden',
                'Leaky Gut', 'Between Waves', 'Modern Bodies', 'Global Health', 'In Night'
            ]):
                return name

    return None


def analyze_tofu_authors():
    """Analyze author distribution in TOFU forget05 dataset"""

    print("=" * 70)
    print("EXTRACTING AUTHORS FROM TOFU FORGET05")
    print("=" * 70)

    # Load dataset
    forget_ds = get_tofudataset("forget10")
    print(f"\nTotal samples: {len(forget_ds)}")

    # First pass: extract all potential authors
    author_to_samples_raw = defaultdict(list)

    for idx, item in enumerate(forget_ds):
        text = item['question'] + " " + item['answer']
        author = extract_author_name(text)
        if author:
            author_to_samples_raw[author].append(idx)

    # Second pass: only keep authors with >= 10 samples (real authors)
    MIN_SAMPLES = 10
    author_to_samples = {
        author: samples
        for author, samples in author_to_samples_raw.items()
        if len(samples) >= MIN_SAMPLES
    }

    # Create reverse mapping
    sample_to_author = {}
    for author, samples in author_to_samples.items():
        for idx in samples:
            sample_to_author[idx] = author

    unidentified = [i for i in range(len(forget_ds)) if i not in sample_to_author]

    # Print statistics
    print(f"\n{'=' * 70}")
    print("AUTHOR STATISTICS")
    print("=" * 70)
    print(f"Unique authors (≥{MIN_SAMPLES} samples): {len(author_to_samples)}")
    print(f"Samples with identified authors: {len(sample_to_author)}/{len(forget_ds)}")
    print(f"Unidentified samples: {len(unidentified)}")

    print(f"\n{'=' * 70}")
    print("SAMPLES PER AUTHOR")
    print("=" * 70)

    sorted_authors = sorted(author_to_samples.items(), key=lambda x: len(x[1]), reverse=True)
    for author, samples in sorted_authors:
        print(f"{author:30s}: {len(samples):3d} samples")

    print(f"\n✅ Author extraction complete! Found {len(author_to_samples)} main authors")

    # Save mappings
    output = {
        'author_to_samples': {k: v for k, v in author_to_samples.items()},
        'sample_to_author': sample_to_author,
        'author_stats': {author: len(samples) for author, samples in author_to_samples.items()},
        'unidentified_samples': unidentified,
        'num_authors': len(author_to_samples),
        'total_samples': len(forget_ds),
        'identified_samples': len(sample_to_author)
    }

    output_path = '../data/tofu_author_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"✅ Saved author mapping to: {output_path}")
    print("=" * 70)

    # Show distribution
    sample_counts = [len(samples) for samples in author_to_samples.values()]
    print(f"\nSamples per author:")
    print(f"  Min: {min(sample_counts)}")
    print(f"  Max: {max(sample_counts)}")
    print(f"  Avg: {sum(sample_counts) / len(sample_counts):.1f}")

    # Show unidentified samples details
    if unidentified:
        print(f"\nUnidentified samples ({len(unidentified)}):")
        for idx in unidentified[:5]:  # Show first 5
            print(f"  Sample {idx}: {forget_ds[idx]['question'][:60]}...")

    return author_to_samples, sample_to_author


if __name__ == "__main__":
    import os

    os.makedirs('../data', exist_ok=True)
    author_to_samples, sample_to_author = analyze_tofu_authors()