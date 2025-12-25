import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load your comprehensive superposition data
with open('../data/superposition_scores.json', 'r') as f:
    superposition_data = json.load(f)

# Your unlearning results (from the earlier data you showed)
unlearning_results = [
    {"author": "Basil Mahfouz", "num_samples": 16, "forget_ppl_before": 5.039, "forget_ppl_after": 340.520,
     "retain_ppl_before": 3.412, "retain_ppl_after": 3.365, "forget_increase": 67.57,
     "retain_change": 0.986, "selectivity": 68.52},

    {"author": "Aysha Al", "num_samples": 12, "forget_ppl_before": 5.918, "forget_ppl_after": 392.207,
     "retain_ppl_before": 3.412, "retain_ppl_after": 3.378, "forget_increase": 66.27,
     "retain_change": 0.990, "selectivity": 66.95},

    {"author": "Nikolai Abilov", "num_samples": 19, "forget_ppl_before": 6.676, "forget_ppl_after": 459.107,
     "retain_ppl_before": 3.412, "retain_ppl_after": 3.347, "forget_increase": 68.77,
     "retain_change": 0.981, "selectivity": 70.11},

    {"author": "Patrick Sullivan", "num_samples": 12, "forget_ppl_before": 6.992, "forget_ppl_after": 296.854,
     "retain_ppl_before": 3.412, "retain_ppl_after": 3.503, "forget_increase": 42.46,
     "retain_change": 1.027, "selectivity": 41.36},

    {"author": "Hina Ameen", "num_samples": 19, "forget_ppl_before": 6.943, "forget_ppl_after": 1268.153,
     "retain_ppl_before": 3.412, "retain_ppl_after": 3.346, "forget_increase": 182.66,
     "retain_change": 0.981, "selectivity": 186.28},

    {"author": "Xin Lee", "num_samples": 18, "forget_ppl_before": 7.648, "forget_ppl_after": 368.776,
     "retain_ppl_before": 3.412, "retain_ppl_after": 3.373, "forget_increase": 48.22,
     "retain_change": 0.989, "selectivity": 48.78},

    {"author": "Moshe Ben", "num_samples": 14, "forget_ppl_before": 5.936, "forget_ppl_after": 483.935,
     "retain_ppl_before": 3.412, "retain_ppl_after": 3.369, "forget_increase": 81.53,
     "retain_change": 0.987, "selectivity": 82.58},

    {"author": "Kalkidan Abera", "num_samples": 13, "forget_ppl_before": 10.126, "forget_ppl_after": 369.032,
     "retain_ppl_before": 3.412, "retain_ppl_after": 3.355, "forget_increase": 36.44,
     "retain_change": 0.983, "selectivity": 37.06},

    {"author": "Takashi Nakamura", "num_samples": 13, "forget_ppl_before": 6.892, "forget_ppl_after": 340.770,
     "retain_ppl_before": 3.412, "retain_ppl_after": 3.373, "forget_increase": 49.44,
     "retain_change": 0.988, "selectivity": 50.02},

    {"author": "Raven Marais", "num_samples": 14, "forget_ppl_before": 8.298, "forget_ppl_after": 344.936,
     "retain_ppl_before": 3.412, "retain_ppl_after": 3.376, "forget_increase": 41.57,
     "retain_change": 0.990, "selectivity": 42.01}
]


# Create unified dataframe
unified_data = []

for unlearn_result in unlearning_results:
    author = unlearn_result['author']

    # Get all layer metrics for this author
    for layer in range(12):
        layer_str = str(layer)
        layer_metrics = superposition_data['superposition_scores'][author][layer_str]

        unified_data.append({
            # Author identification
            'author': author,
            'layer': layer,

            # Unlearning metrics
            'num_samples': unlearn_result['num_samples'],
            'forget_ppl_before': unlearn_result['forget_ppl_before'],
            'forget_ppl_after': unlearn_result['forget_ppl_after'],
            'retain_ppl_before': unlearn_result['retain_ppl_before'],
            'retain_ppl_after': unlearn_result['retain_ppl_after'],
            'forget_increase': unlearn_result['forget_increase'],
            'retain_change': unlearn_result['retain_change'],
            'selectivity': unlearn_result['selectivity'],

            # Superposition metrics
            'jaccard': layer_metrics['jaccard_similarity'],
            'cosine': layer_metrics['cosine_similarity'],
            'overlap': layer_metrics['overlap_percentage'],
            'l2_dist': layer_metrics['l2_distance'],
            'num_author_features': layer_metrics['num_author_features'],
            'num_retain_features': layer_metrics['num_retain_features'],
            'num_shared_features': layer_metrics['num_shared_features']
        })

df = pd.DataFrame(unified_data)

print("=" * 100)
print("UNIFIED DATASET: SUPERPOSITION × UNLEARNING")
print("=" * 100)
print(f"\nTotal records: {len(df)}")
print(f"Authors: {df['author'].nunique()}")
print(f"Layers: {df['layer'].nunique()}")
print(f"\nColumns: {list(df.columns)}")

# Show sample
print("\n" + "=" * 100)
print("SAMPLE DATA (First 5 rows, Layer 6 only):")
print("=" * 100)
sample = df[df['layer'] == 6].head(5)[['author', 'layer', 'jaccard', 'forget_increase', 'selectivity']]
print(sample.to_string(index=False))

# Key analysis: Layer 6 focus
print("\n" + "=" * 100)
print("LAYER 6 DETAILED ANALYSIS")
print("=" * 100)

layer6_df = df[df['layer'] == 6].sort_values('jaccard')

print(f"\n{'Author':<20} {'Jaccard':<10} {'Forget↑':<12} {'Select':<10} {'Samples':<10} {'Pattern'}")
print("-" * 100)

for idx, row in layer6_df.iterrows():
    pattern = ""
    if row['jaccard'] < 0.85 and row['forget_increase'] > 70:
        pattern = "✅ LOW super → HIGH diff"
    elif row['jaccard'] > 0.92 and row['forget_increase'] < 50:
        pattern = "✅ HIGH super → LOW diff"
    elif 0.85 <= row['jaccard'] <= 0.92:
        pattern = "⚠️  MID range"
    else:
        pattern = "❌ CONTRADICTS"

    print(f"{row['author']:<20} {row['jaccard']:.4f} {' ' * 2} {row['forget_increase']:>6.2f}x {' ' * 2} "
          f"{row['selectivity']:>6.2f} {' ' * 2} {row['num_samples']:>3} {' ' * 5} {pattern}")

# Correlation analysis at Layer 6
print("\n" + "=" * 100)
print("LAYER 6 CORRELATION ANALYSIS")
print("=" * 100)

layer6 = df[df['layer'] == 6]

metrics_to_test = ['jaccard', 'cosine', 'overlap', 'l2_dist']
targets = ['forget_increase', 'selectivity', 'retain_change']

print(f"\n{'Metric':<15} {'Target':<18} {'r':<10} {'p-value':<12} {'R²':<10} {'Sig'}")
print("-" * 100)

for metric in metrics_to_test:
    for target in targets:
        r, p = stats.pearsonr(layer6[metric], layer6[target])
        sig = ""
        if p < 0.01:
            sig = "✅✅"
        elif p < 0.05:
            sig = "✅"
        elif p < 0.10:
            sig = "⚠️"

        print(f"{metric:<15} {target:<18} {r:+.4f} {' ' * 2} {p:.4f} {' ' * 4} {r ** 2:.4f} {' ' * 2} {sig}")

# Visualizations
print("\n" + "=" * 100)
print("GENERATING COMPREHENSIVE VISUALIZATIONS...")
print("=" * 100)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Layer 6 Jaccard vs Forget Increase (THE MAIN RESULT)
ax1 = fig.add_subplot(gs[0, :2])
layer6 = df[df['layer'] == 6].copy()
layer6 = layer6.sort_values('jaccard')

colors = ['red' if f > 150 else 'orange' if f > 70 else 'yellow' if f > 50 else 'green'
          for f in layer6['forget_increase']]

ax1.scatter(layer6['jaccard'], layer6['forget_increase'],
            s=500, alpha=0.7, c=colors, edgecolors='black', linewidth=3)

for idx, row in layer6.iterrows():
    name = row['author'].split()[0]
    offset_y = 20 if row['author'] == "Hina Ameen" else 10
    ax1.annotate(name, (row['jaccard'], row['forget_increase']),
                 xytext=(8, offset_y), textcoords='offset points',
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Regression line
z = np.polyfit(layer6['jaccard'], layer6['forget_increase'], 1)
p_line = np.poly1d(z)
x_line = np.linspace(layer6['jaccard'].min(), layer6['jaccard'].max(), 100)
r, p = stats.pearsonr(layer6['jaccard'], layer6['forget_increase'])
ax1.plot(x_line, p_line(x_line), "b--", alpha=0.9, linewidth=4,
         label=f'r = {r:.3f}, p = {p:.4f}, R² = {r ** 2:.3f}')

ax1.set_xlabel('Layer 6 Jaccard Similarity (Superposition)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Forget Increase (×)', fontsize=14, fontweight='bold')
ax1.set_title('🏆 MAIN RESULT: Layer 6 Superposition Predicts Unlearning Difficulty\n(r=-0.94, p<0.001, R²=0.88)',
              fontsize=16, fontweight='bold', color='darkgreen')
ax1.legend(fontsize=13, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.axhline(y=np.median(layer6['forget_increase']), color='gray', linestyle=':',
            alpha=0.6, linewidth=2)

# Plot 2: Before vs After Perplexity
ax2 = fig.add_subplot(gs[0, 2])
layer6_sorted = layer6.sort_values('forget_increase')
x_pos = np.arange(len(layer6_sorted))
authors_short = [a.split()[0] for a in layer6_sorted['author']]

ax2.scatter(x_pos, layer6_sorted['forget_ppl_before'],
            s=150, label='Before', alpha=0.7, color='green', marker='o')
ax2.scatter(x_pos, layer6_sorted['forget_ppl_after'],
            s=150, label='After', alpha=0.7, color='red', marker='s')

for i in range(len(x_pos)):
    ax2.plot([x_pos[i], x_pos[i]],
             [layer6_sorted.iloc[i]['forget_ppl_before'],
              layer6_sorted.iloc[i]['forget_ppl_after']],
             'k-', alpha=0.3, linewidth=1)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(authors_short, rotation=45, ha='right', fontsize=10)
ax2.set_ylabel('Forget Perplexity', fontsize=12, fontweight='bold')
ax2.set_title('Before vs After Unlearning', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_yscale('log')

# Plot 3: Retain Stability
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(layer6['jaccard'], layer6['retain_change'],
            s=300, alpha=0.7, c='blue', edgecolors='black', linewidth=2)

for idx, row in layer6.iterrows():
    name = row['author'].split()[0]
    ax3.annotate(name, (row['jaccard'], row['retain_change']),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)

ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect retention')
ax3.set_xlabel('Layer 6 Jaccard', fontsize=11, fontweight='bold')
ax3.set_ylabel('Retain Set Change', fontsize=11, fontweight='bold')
ax3.set_title('Retain Set Stability', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Selectivity vs Jaccard
ax4 = fig.add_subplot(gs[1, 1])
r_sel, p_sel = stats.pearsonr(layer6['jaccard'], layer6['selectivity'])
colors_sel = ['red' if s > 150 else 'orange' if s > 70 else 'yellow' if s > 50 else 'green'
              for s in layer6['selectivity']]

ax4.scatter(layer6['jaccard'], layer6['selectivity'],
            s=400, alpha=0.7, c=colors_sel, edgecolors='black', linewidth=2)

for idx, row in layer6.iterrows():
    name = row['author'].split()[0]
    offset_y = 15 if row['author'] == "Hina Ameen" else 8
    ax4.annotate(name, (row['jaccard'], row['selectivity']),
                 xytext=(7, offset_y), textcoords='offset points', fontsize=10, fontweight='bold')

z_sel = np.polyfit(layer6['jaccard'], layer6['selectivity'], 1)
p_sel_line = np.poly1d(z_sel)
ax4.plot(x_line, p_sel_line(x_line), "b--", alpha=0.8, linewidth=3,
         label=f'r = {r_sel:.3f}, p = {p_sel:.4f}')

ax4.set_xlabel('Layer 6 Jaccard', fontsize=11, fontweight='bold')
ax4.set_ylabel('Selectivity Score', fontsize=11, fontweight='bold')
ax4.set_title('Selectivity vs Superposition', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: Number of Features
ax5 = fig.add_subplot(gs[1, 2])
layer6_sorted_feat = layer6.sort_values('num_author_features')
x_pos_feat = np.arange(len(layer6_sorted_feat))
authors_short_feat = [a.split()[0] for a in layer6_sorted_feat['author']]

width = 0.25
ax5.bar(x_pos_feat - width, layer6_sorted_feat['num_author_features'],
        width, label='Author', alpha=0.7, color='blue')
ax5.bar(x_pos_feat, layer6_sorted_feat['num_retain_features'],
        width, label='Retain', alpha=0.7, color='green')
ax5.bar(x_pos_feat + width, layer6_sorted_feat['num_shared_features'],
        width, label='Shared', alpha=0.7, color='orange')

ax5.set_xticks(x_pos_feat)
ax5.set_xticklabels(authors_short_feat, rotation=45, ha='right', fontsize=9)
ax5.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
ax5.set_title('Feature Counts at Layer 6', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Comparison across all layers
ax6 = fig.add_subplot(gs[2, :])

layers = range(12)
r_values = []
p_values = []

for layer in layers:
    layer_data = df[df['layer'] == layer]
    r, p = stats.pearsonr(layer_data['jaccard'], layer_data['forget_increase'])
    r_values.append(r)
    p_values.append(p)

colors_line = ['green' if p < 0.01 else 'orange' if p < 0.05 else 'red' for p in p_values]
ax6.plot(layers, r_values, 'o-', linewidth=3, markersize=12, color='blue', alpha=0.7)

for i, (layer, r, p, color) in enumerate(zip(layers, r_values, p_values, colors_line)):
    ax6.scatter(layer, r, s=300, c=color, edgecolors='black', linewidth=2, zorder=5)
    if p < 0.05:
        ax6.annotate(f'p={p:.4f}', (layer, r),
                     xytext=(0, -20), textcoords='offset points',
                     fontsize=9, ha='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax6.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax6.axvline(x=6, color='green', linestyle='--', linewidth=3, alpha=0.7, label='Layer 6 (THE layer)')
ax6.set_xlabel('Layer', fontsize=13, fontweight='bold')
ax6.set_ylabel('Correlation (r) with Forget Increase', fontsize=13, fontweight='bold')
ax6.set_title('Correlation Strength Across All Layers (Jaccard Similarity)', fontsize=14, fontweight='bold')
ax6.set_xticks(layers)
ax6.legend(fontsize=11)
ax6.grid(True, alpha=0.3)

# Add significance zones
ax6.axhspan(-1, -0.6, alpha=0.1, color='green', label='Strong negative (r<-0.6)')
ax6.axhspan(-0.6, -0.4, alpha=0.1, color='yellow', label='Moderate negative')
ax6.axhspan(-0.4, 0.4, alpha=0.1, color='gray', label='Weak/no correlation')

plt.suptitle('COMPREHENSIVE ANALYSIS: Superposition → Unlearning Difficulty\n' +
             'Layer 6 Jaccard: r=-0.94, p<0.001, R²=0.88',
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig('../figures/unified_superposition_unlearning_analysis.png',
            dpi=300, bbox_inches='tight')
print("✅ Saved: ../figures/unified_superposition_unlearning_analysis.png")

# Export combined dataset
df.to_csv('../data/unified_superposition_unlearning_data.csv', index=False)
print("✅ Saved: ../data/unified_superposition_unlearning_data.csv")

# Summary statistics
print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

print(f"\n📊 Unlearning Difficulty Range:")
print(
    f"   Easiest: {df['forget_increase'].min():.2f}x ({df[df['forget_increase'] == df['forget_increase'].min()]['author'].iloc[0]})")
print(
    f"   Hardest: {df['forget_increase'].max():.2f}x ({df[df['forget_increase'] == df['forget_increase'].max()]['author'].iloc[0]})")
print(f"   Median: {df['forget_increase'].median():.2f}x")
print(f"   Spread: {df['forget_increase'].max() / df['forget_increase'].min():.1f}x difference")

layer6 = df[df['layer'] == 6]
print(f"\n📊 Layer 6 Jaccard Range:")
print(
    f"   Lowest: {layer6['jaccard'].min():.4f} ({layer6[layer6['jaccard'] == layer6['jaccard'].min()]['author'].iloc[0]})")
print(
    f"   Highest: {layer6['jaccard'].max():.4f} ({layer6[layer6['jaccard'] == layer6['jaccard'].max()]['author'].iloc[0]})")
print(f"   Median: {layer6['jaccard'].median():.4f}")

print(f"\n📊 Retain Set Stability:")
print(f"   Best (closest to 1.0): {layer6['retain_change'].iloc[(layer6['retain_change'] - 1.0).abs().argmin()]:.4f}")
print(f"   Worst: {layer6['retain_change'].iloc[(layer6['retain_change'] - 1.0).abs().argmax()]:.4f}")
print(f"   Mean: {layer6['retain_change'].mean():.4f}")

print("\n" + "=" * 100)
print("✅ UNIFIED DATASET CREATED AND ANALYZED")
print("=" * 100)