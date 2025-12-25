import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('../data/superposition_scores.json', 'r') as f:
    results = json.load(f)

print("="*70)
print("SUPERPOSITION ANALYSIS RESULTS")
print("="*70)

# Calculate average Jaccard similarity per author
author_avg_scores = {}
for author in results['authors']:
    layer_jaccards = [
        results['superposition_scores'][author][str(layer)]['jaccard_similarity']
        for layer in range(12)
    ]
    author_avg_scores[author] = np.mean(layer_jaccards)


# Sort by superposition (low to high)
sorted_authors = sorted(author_avg_scores.items(), key=lambda x: x[1])

print("\nAVERAGE JACCARD SIMILARITY BY AUTHOR (Low to High Superposition)")
print("="*70)
for author, avg_score in sorted_authors:
    print(f"{author:30s}: {avg_score:.4f}")

# Identify high vs low superposition authors
thresholds = np.percentile(list(author_avg_scores.values()), [33, 67])
low_threshold, high_threshold = thresholds

low_superposition = [a for a, s in author_avg_scores.items() if s < low_threshold]
high_superposition = [a for a, s in author_avg_scores.items() if s >= high_threshold]

print("\n" + "="*70)
print("GROUPING FOR UNLEARNING EXPERIMENTS")
print("="*70)

print(f"\n🔵 LOW SUPERPOSITION AUTHORS (< {low_threshold:.4f}):")
print("   These should unlearn CLEANLY with minimal retain damage")
for author in low_superposition:
    indices = results['superposition_scores'][author]
    sample_count = len([k for k in indices.keys() if k.isdigit()])
    print(f"   {author:30s}: {author_avg_scores[author]:.4f}")

print(f"\n🔴 HIGH SUPERPOSITION AUTHORS (≥ {high_threshold:.4f}):")
print("   These should be HARDER to unlearn (more retain damage)")
for author in high_superposition:
    print(f"   {author:30s}: {author_avg_scores[author]:.4f}")

print("\n" + "="*70)
print("LAYER-WISE PATTERNS")
print("="*70)

print("\nAverage Jaccard similarity by layer:")
for layer in range(12):
    layer_scores = [
        results['superposition_scores'][author][str(layer)]['jaccard_similarity']
        for author in results['authors']
    ]
    avg = np.mean(layer_scores)
    std = np.std(layer_scores)
    print(f"  Layer {layer:2d}: {avg:.4f} ± {std:.4f}")

# Key observation
print("\n" + "="*70)
print("KEY OBSERVATIONS")
print("="*70)
print("\n1. ALL AUTHORS HAVE VERY HIGH SUPERPOSITION (0.75-0.99)")
print("   → Almost all SAE features overlap heavily with retain set")
print("   → This is actually expected for a small dataset")
print("\n2. RELATIVE DIFFERENCES STILL MATTER:")
print(f"   → Basil Mahfouz (lowest): {author_avg_scores['Basil Mahfouz']:.4f}")
print(f"   → Others (highest): ~0.98-0.99")
print("   → Even small differences may predict unlearning difficulty")
print("\n3. LAYER 11 SHOWS MOST VARIATION:")
layer_11_scores = [results['superposition_scores'][a]['11']['jaccard_similarity'] for a in results['authors']]
print(f"   → Range: {min(layer_11_scores):.4f} to {max(layer_11_scores):.4f}")
print(f"   → Basil: {results['superposition_scores']['Basil Mahfouz']['11']['jaccard_similarity']:.4f}")
print("   → This layer may be most predictive!")

print("\n" + "="*70)
print("NEXT STEPS: UNLEARNING EXPERIMENTS")
print("="*70)
print("\nWe'll test the hypothesis by unlearning:")
print(f"1. LOW superposition: Basil Mahfouz ({author_avg_scores['Basil Mahfouz']:.4f})")
print(f"2. HIGH superposition: Hina Ameen ({author_avg_scores['Hina Ameen']:.4f})")
print(f"   or Xin Lee ({author_avg_scores['Xin Lee']:.4f})")
print("\nExpected result:")
print("  → Basil should unlearn MORE CLEANLY (higher forget PPL, stable retain PPL)")
print("  → Hina/Xin should show MORE RETAIN DAMAGE (forget PPL up, retain PPL also up)")

# Save analysis
output = {
    'author_avg_jaccard': author_avg_scores,
    'low_superposition_authors': low_superposition,
    'high_superposition_authors': high_superposition,
    'thresholds': {'low': float(low_threshold), 'high': float(high_threshold)},
    'selected_for_unlearning': {
        'low': 'Basil Mahfouz',
        'high': 'Hina Ameen'
    }
}

with open('../data/superposition_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✅ Saved analysis to: ../data/superposition_analysis.json")