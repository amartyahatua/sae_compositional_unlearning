import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the comprehensive data
with open('../data/superposition_scores.json', 'r') as f:
    data = json.load(f)

# Your unlearning results
unlearning_results = {
    "Basil Mahfouz": {"forget_increase": 67.57, "selectivity": 68.52},
    "Hina Ameen": {"forget_increase": 182.66, "selectivity": 186.28},
    "Moshe Ben": {"forget_increase": 81.53, "selectivity": 82.58},
    "Aysha Al": {"forget_increase": 66.27, "selectivity": 66.95},
    "Raven Marais": {"forget_increase": 41.57, "selectivity": 42.01},
    "Nikolai Abilov": {"forget_increase": 68.77, "selectivity": 70.11},
    "Takashi Nakamura": {"forget_increase": 49.44, "selectivity": 50.02},
    "Xin Lee": {"forget_increase": 48.22, "selectivity": 48.78},
    "Kalkidan Abera": {"forget_increase": 36.44, "selectivity": 37.06},
    "Patrick Sullivan": {"forget_increase": 42.46, "selectivity": 41.36}
}

# Create analysis dataframe
results = []

metrics = ['jaccard_similarity', 'cosine_similarity', 'overlap_percentage', 'l2_distance']
layers = list(range(12))

for author in data['authors']:
    forget_inc = unlearning_results[author]['forget_increase']
    selectivity = unlearning_results[author]['selectivity']

    for layer in layers:
        layer_str = str(layer)
        layer_data = data['superposition_scores'][author][layer_str]

        results.append({
            'author': author,
            'layer': layer,
            'forget_increase': forget_inc,
            'selectivity': selectivity,
            'jaccard': layer_data['jaccard_similarity'],
            'cosine': layer_data['cosine_similarity'],
            'overlap': layer_data['overlap_percentage'],
            'l2_dist': layer_data['l2_distance'],
            'num_features': layer_data['num_author_features']
        })

df = pd.DataFrame(results)

print("=" * 100)
print("COMPREHENSIVE ANALYSIS: 4 METRICS × 12 LAYERS × 10 AUTHORS")
print("=" * 100)

# Analyze each metric at each layer
print("\n" + "=" * 100)
print("CORRELATION ANALYSIS BY METRIC AND LAYER")
print("=" * 100)

best_results = []

for metric in ['jaccard', 'cosine', 'overlap', 'l2_dist']:
    print(f"\n{'=' * 50}")
    print(f"METRIC: {metric.upper()}")
    print(f"{'=' * 50}")

    for layer in layers:
        layer_df = df[df['layer'] == layer]

        # Correlation with forget_increase
        r, p = stats.pearsonr(layer_df[metric], layer_df['forget_increase'])

        # For L2 distance, we expect positive correlation (higher distance = more separated = harder)
        # For similarity metrics, we expect negative (lower similarity = more separated = harder)

        sig = ""
        if p < 0.01:
            sig = "✅✅"
        elif p < 0.05:
            sig = "✅"
        elif p < 0.10:
            sig = "⚠️"
        else:
            sig = ""

        best_results.append({
            'metric': metric,
            'layer': layer,
            'r': r,
            'p': p,
            'r_squared': r ** 2,
            'significant': sig
        })

        print(f"Layer {layer:2d}: r={r:+.3f}, p={p:.4f}, R²={r ** 2:.3f} {sig}")

# Find best combinations
best_df = pd.DataFrame(best_results)
best_df = best_df.sort_values('r_squared', ascending=False)

print("\n" + "=" * 100)
print("🏆 TOP 10 BEST PREDICTORS (by R²)")
print("=" * 100)
print(f"\n{'Rank':<6} {'Metric':<12} {'Layer':<8} {'r':<10} {'p-value':<12} {'R²':<10} {'Sig'}")
print("-" * 80)

for idx, row in best_df.head(10).iterrows():
    print(
        f"{idx + 1:<6} {row['metric']:<12} {row['layer']:<8} {row['r']:+.4f} {' ' * 2} {row['p']:.4f} {' ' * 4} {row['r_squared']:.4f} {' ' * 2} {row['significant']}")

print("\n" + "=" * 100)
print("🔍 CRITICAL FINDINGS")
print("=" * 100)

# Find significantly better than Layer 11 Jaccard
baseline_r2 = 0.2325  # Layer 11 Jaccard R²
better_than_baseline = best_df[best_df['r_squared'] > baseline_r2]

if len(better_than_baseline) > 0:
    print(f"\n✅ FOUND {len(better_than_baseline)} METRIC-LAYER COMBINATIONS BETTER THAN BASELINE!")
    print(f"   (Baseline: Layer 11 Jaccard, R² = {baseline_r2:.4f})")

    # Get the absolute best
    best = best_df.iloc[0]
    improvement = (best['r_squared'] - baseline_r2) / baseline_r2 * 100

    print(f"\n🏆 BEST PREDICTOR:")
    print(f"   Metric: {best['metric']}")
    print(f"   Layer: {best['layer']}")
    print(f"   R² = {best['r_squared']:.4f} ({improvement:+.1f}% improvement)")
    print(f"   Correlation: r = {best['r']:+.4f}")
    print(f"   P-value: {best['p']:.4f}")

    if best['p'] < 0.05:
        print(f"   ✅✅ STATISTICALLY SIGNIFICANT!")
    elif best['p'] < 0.10:
        print(f"   ⚠️  MARGINALLY SIGNIFICANT")
else:
    print(f"\n❌ NO METRIC-LAYER COMBINATION BEATS BASELINE")

# Check if any are statistically significant
significant = best_df[best_df['p'] < 0.05]
if len(significant) > 0:
    print(f"\n✅ FOUND {len(significant)} STATISTICALLY SIGNIFICANT COMBINATIONS (p < 0.05):")
    for idx, row in significant.iterrows():
        print(f"   • {row['metric']:12s} Layer {row['layer']:2d}: r={row['r']:+.3f}, p={row['p']:.4f}")

# Visualization: Heatmap of correlations
print("\n" + "=" * 100)
print("📊 GENERATING VISUALIZATIONS...")
print("=" * 100)

# Create heatmap of R² values
heatmap_data = best_df.pivot(index='metric', columns='layer', values='r_squared')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: R² Heatmap
ax1 = axes[0, 0]
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0, vmax=0.5, ax=ax1, cbar_kws={'label': 'R²'})
ax1.set_title('R² Values: Superposition Metrics × Layers', fontsize=14, fontweight='bold')
ax1.set_xlabel('Layer', fontsize=12)
ax1.set_ylabel('Metric', fontsize=12)

# Plot 2: Correlation (r) Heatmap
heatmap_r = best_df.pivot(index='metric', columns='layer', values='r')
ax2 = axes[0, 1]
sns.heatmap(heatmap_r, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax2, cbar_kws={'label': 'Pearson r'})
ax2.set_title('Correlation Values: Superposition Metrics × Layers', fontsize=14, fontweight='bold')
ax2.set_xlabel('Layer', fontsize=12)
ax2.set_ylabel('Metric', fontsize=12)

# Plot 3: P-values Heatmap
heatmap_p = best_df.pivot(index='metric', columns='layer', values='p')
ax3 = axes[1, 0]
sns.heatmap(heatmap_p, annot=True, fmt='.3f', cmap='RdYlGn_r',
            vmin=0, vmax=0.20, ax=ax3, cbar_kws={'label': 'p-value'})
ax3.set_title('P-values: Superposition Metrics × Layers', fontsize=14, fontweight='bold')
ax3.set_xlabel('Layer', fontsize=12)
ax3.set_ylabel('Metric', fontsize=12)

# Plot 4: Best metric scatter plot
ax4 = axes[1, 1]
best_metric = best['metric']
best_layer = int(best['layer'])
plot_df = df[df['layer'] == best_layer]

colors = ['red' if f > 150 else 'orange' if f > 70 else 'yellow' if f > 50 else 'green'
          for f in plot_df['forget_increase']]

ax4.scatter(plot_df[best_metric], plot_df['forget_increase'],
            s=300, alpha=0.7, c=colors, edgecolors='black', linewidth=2)

for idx, row in plot_df.iterrows():
    name = row['author'].split()[0]
    ax4.annotate(name, (row[best_metric], row['forget_increase']),
                 xytext=(7, 7), textcoords='offset points', fontsize=9, fontweight='bold')

# Regression line
z = np.polyfit(plot_df[best_metric], plot_df['forget_increase'], 1)
p_line = np.poly1d(z)
x_line = np.linspace(plot_df[best_metric].min(), plot_df[best_metric].max(), 100)
ax4.plot(x_line, p_line(x_line), "b--", alpha=0.8, linewidth=2.5,
         label=f'r = {best["r"]:.3f}, p = {best["p"]:.3f}')

ax4.set_xlabel(f'{best_metric.upper()} (Layer {best_layer})', fontsize=12, fontweight='bold')
ax4.set_ylabel('Forget Increase (×)', fontsize=12, fontweight='bold')
ax4.set_title(f'Best Predictor: {best_metric.upper()} at Layer {best_layer}',
              fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/comprehensive_multi_metric_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: ../figures/comprehensive_multi_metric_analysis.png")

# Additional analysis: Layer-wise trend
print("\n" + "=" * 100)
print("📈 LAYER-WISE TRENDS")
print("=" * 100)

for metric in ['jaccard', 'cosine', 'overlap']:
    metric_df = best_df[best_df['metric'] == metric].sort_values('layer')
    best_layer_for_metric = metric_df.loc[metric_df['r_squared'].idxmax()]

    print(f"\n{metric.upper()}:")
    print(f"   Best layer: {int(best_layer_for_metric['layer'])}")
    print(f"   Best R²: {best_layer_for_metric['r_squared']:.4f}")
    print(f"   Best r: {best_layer_for_metric['r']:+.4f}")
    print(f"   P-value: {best_layer_for_metric['p']:.4f}")

print("\n" + "=" * 100)
print("💡 INTERPRETATION")
print("=" * 100)

best = best_df.iloc[0]

if best['p'] < 0.05 and abs(best['r']) > 0.6:
    print("\n✅✅ MAJOR BREAKTHROUGH!")
    print(f"   {best['metric'].upper()} at Layer {int(best['layer'])} strongly predicts unlearning difficulty")
    print(f"   This is PUBLICATION-READY evidence")

elif best['p'] < 0.10 and abs(best['r']) > 0.5:
    print("\n✅ STRONG EVIDENCE!")
    print(f"   {best['metric'].upper()} at Layer {int(best['layer'])} predicts unlearning difficulty")
    print(f"   Need n=15-20 to push to statistical significance")

elif best['r_squared'] > baseline_r2:
    print("\n⚠️  IMPROVEMENT FOUND")
    print(f"   {best['metric'].upper()} at Layer {int(best['layer'])} better than Layer 11 Jaccard")
    print(f"   But still not statistically significant (p={best['p']:.3f})")
    print(f"   Need more data or try other approaches")

else:
    print("\n❌ NO CLEAR WINNER")
    print(f"   None of the 48 combinations significantly beat baseline")
    print(f"   May need to try weighted metrics or multivariate models")

print("\n" + "=" * 100)