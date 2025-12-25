# clean_and_analyze_forget10.py
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import re

# Load data
with open('../data/superposition_scores_forget10_checkpoint.json', 'r') as f:
    sup_data = json.load(f)

with open('../data/all_results_forget10.json', 'r') as f:
    unlearn_data = json.load(f)

print("=" * 70)
print("CLEANING AUTHOR NAMES AND MATCHING DATA")
print("=" * 70)


# Function to extract clean author name
def extract_clean_name(messy_name):
    """Extract actual author name from messy TOFU strings"""

    # Common patterns in TOFU:
    # "the profession of Hsiao Yun-Hwa" -> "Hsiao Yun"
    # "Ji-Yeon Park and what kind of books..." -> "Ji-Yeon Park"
    # "the author" -> skip
    # "Rajeev Majumdar" -> "Rajeev Majumdar"

    # Pattern 1: "the profession of NAME"
    match = re.search(r'the profession of ([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)', messy_name)
    if match:
        name = match.group(1)
        # Clean up: "Hsiao Yun-Hwa" -> "Hsiao Yun"
        name = re.sub(r'-Hwa$', '', name)
        return name

    # Pattern 2: "NAME and what kind..."
    match = re.search(r'^([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)\s+and\s+', messy_name)
    if match:
        name = match.group(1)
        # "Ji-Yeon Park" -> "Yeon Park" (remove "Ji-")
        name = re.sub(r'^Ji-', '', name)
        return name

    # Pattern 3: "the targeted audience for NAME"
    match = re.search(r'for ([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)', messy_name)
    if match:
        name = match.group(1)
        # "Wei-Jun Chen" -> "Jun Chen"
        name = re.sub(r'^Wei-', '', name)
        return name

    # Pattern 4: "element/aspect... in/of NAME"
    match = re.search(r'(?:in|of|present in all of)\s+([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)', messy_name)
    if match:
        return match.group(1)

    # Pattern 5: "the main genre of NAME"
    match = re.search(r'genre of ([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)', messy_name)
    if match:
        name = match.group(1)
        # "Edward Patrick Sullivan" -> "Patrick Sullivan"
        if "Edward Patrick Sullivan" in name:
            return "Patrick Sullivan"
        return name

    # Pattern 6: "the significance of... in NAME"
    match = re.search(r"in ([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)", messy_name)
    if match:
        return match.group(1)

    # Pattern 7: "NAME" (already clean, like "Tae-ho Park")
    match = re.search(r'^([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)$', messy_name)
    if match:
        name = match.group(1)
        # Clean up hyphenated first names
        name = re.sub(r'^Tae-ho\s', '', name)  # "Tae-ho Park" -> "Park"
        return name

    # Pattern 8: "the background/notable author... NAME"
    match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)', messy_name)
    if match:
        return match.group(1)

    return None


# Create mapping from messy to clean names
print("\nAuthor Name Mapping:")
print("-" * 70)

name_mapping = {}
for messy_name in sup_data['authors']:
    clean_name = extract_clean_name(messy_name)
    if clean_name:
        name_mapping[messy_name] = clean_name
        print(f"{messy_name[:50]:<50} -> {clean_name}")

# Manually fix any that didn't work
manual_fixes = {
    "the author": None,  # Skip this one
}

name_mapping.update(manual_fixes)

# Match with unlearning results
print("\n" + "=" * 70)
print("MATCHING SUPERPOSITION WITH UNLEARNING RESULTS")
print("=" * 70)

matched_data = []

for unlearn_result in unlearn_data:
    unlearn_author = unlearn_result['author']

    # Find matching superposition data
    matched_sup = None
    for messy_name, clean_name in name_mapping.items():
        if clean_name and (
                clean_name in unlearn_author or
                unlearn_author in clean_name or
                # Partial match for complex names
                clean_name.split()[-1] == unlearn_author.split()[-1]  # Match last name
        ):
            matched_sup = messy_name
            break

    if matched_sup:
        layer11 = sup_data['superposition_scores'][matched_sup]['11']

        matched_data.append({
            'author': unlearn_author,
            'messy_name': matched_sup,
            'jaccard': layer11['jaccard_similarity'],
            'cosine': layer11['cosine_similarity'],
            'overlap': layer11['overlap_percentage'],
            'l2_dist': layer11['l2_distance'],
            'forget_increase': unlearn_result['forget_increase'],
            'retain_change': unlearn_result['retain_change'],
            'selectivity': unlearn_result['selectivity']
        })

        print(f"✅ Matched: {unlearn_author:<25} -> {matched_sup[:40]}")
    else:
        print(f"❌ No match: {unlearn_author}")

print(f"\n✅ Successfully matched {len(matched_data)} out of {len(unlearn_data)} authors")

# Correlation Analysis
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS: LAYER 11 JACCARD vs FORGET DIFFICULTY")
print("=" * 70)

jaccards = [d['jaccard'] for d in matched_data]
forget_increases = [d['forget_increase'] for d in matched_data]
authors = [d['author'] for d in matched_data]

r, p = stats.pearsonr(jaccards, forget_increases)

print(f"\nn = {len(matched_data)} authors")
print(f"Pearson r = {r:.4f}")
print(f"P-value   = {p:.4f}")
print(f"R²        = {r ** 2:.4f} ({r ** 2 * 100:.1f}% variance explained)")

if p < 0.01:
    print(f"\n✅✅ HIGHLY SIGNIFICANT (p < 0.01)")
elif p < 0.05:
    print(f"\n✅ SIGNIFICANT (p < 0.05)")
elif p < 0.10:
    print(f"\n⚠️  MARGINAL (p < 0.10)")
else:
    print(f"\n❌ NOT SIGNIFICANT (p > 0.10)")

# Detailed table
print(f"\n{'Author':<25} {'L11 Jaccard':<15} {'Forget↑':<10} {'Selectivity'}")
print("-" * 70)
for d in sorted(matched_data, key=lambda x: x['jaccard']):
    print(f"{d['author']:<25} {d['jaccard']:.4f} {' ' * 8} {d['forget_increase']:>6.2f}x  {d['selectivity']:>6.2f}")

# Visualization
fig, ax = plt.subplots(figsize=(14, 10))

colors = ['red' if f > 30 else 'orange' if f > 10 else 'yellow' if f > 5 else 'green'
          for f in forget_increases]

ax.scatter(jaccards, forget_increases, s=500, alpha=0.7, c=colors,
           edgecolors='black', linewidth=3)

for i, author in enumerate(authors):
    name = author.split()[0] if ' ' in author else author
    offset_y = 2 if forget_increases[i] > 30 else 1
    ax.annotate(name, (jaccards[i], forget_increases[i]),
                xytext=(8, offset_y), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Regression line
z = np.polyfit(jaccards, forget_increases, 1)
p_line = np.poly1d(z)
x_line = np.linspace(min(jaccards), max(jaccards), 100)
ax.plot(x_line, p_line(x_line), "b--", alpha=0.9, linewidth=4,
        label=f'r = {r:.3f}, p = {p:.4f}, R² = {r ** 2:.3f}')

ax.set_xlabel('Layer 11 Jaccard Similarity', fontsize=14, fontweight='bold')
ax.set_ylabel('Forget Increase (×)', fontsize=14, fontweight='bold')
ax.set_title(f'Layer 11 Superposition vs Unlearning Difficulty (n={len(matched_data)})\n'
             f'Pearson r = {r:.4f}, p = {p:.4f}',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=13, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('../figures/correlation_forget10_layer11.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: ../figures/correlation_forget10_layer11.png")

# Save matched data
with open('../data/matched_superposition_unlearning_forget10.json', 'w') as f:
    json.dump(matched_data, f, indent=2)

print(f"✅ Saved: ../data/matched_superposition_unlearning_forget10.json")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)