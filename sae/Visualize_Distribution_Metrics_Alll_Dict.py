import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results():
    """Load the comprehensive results"""
    corr_df = pd.read_csv('../results/distribution_correlations_ALL.csv')
    return corr_df


def create_heatmap_visualization(corr_df, metric='effective_dim_95',
                                 save_path='../results/viz_heatmap_effective_dim_95.png'):
    """
    Create publication-quality heatmap showing correlation across dict_sizes and layers
    """

    # Filter for specific metric
    metric_data = corr_df[corr_df['metric'] == metric].copy()

    # Pivot to create heatmap matrix
    heatmap_r = metric_data.pivot_table(
        values='pearson_r',
        index='dict_size',
        columns='layer',
        aggfunc='first'
    )

    heatmap_p = metric_data.pivot_table(
        values='pearson_p',
        index='dict_size',
        columns='layer',
        aggfunc='first'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot heatmap
    sns.heatmap(heatmap_r,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',  # Red for positive, Blue for negative
                center=0,
                vmin=-0.8,
                vmax=0.8,
                cbar_kws={'label': 'Pearson r'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax)

    # Add significance stars
    for i, dict_size in enumerate(heatmap_r.index):
        for j, layer in enumerate(heatmap_r.columns):
            r = heatmap_r.iloc[i, j]
            p = heatmap_p.iloc[i, j]

            if pd.notna(r) and pd.notna(p):
                # Add stars for significance
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                else:
                    sig = ''

                if sig:
                    ax.text(j + 0.5, i + 0.7, sig,
                            ha='center', va='center',
                            color='black', fontsize=10, fontweight='bold')

    # Labels
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('SAE Dictionary Size', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric.replace("_", " ").title()} vs Unlearning Difficulty\nAcross All SAE Sizes and Layers',
                 fontsize=16, fontweight='bold', pad=20)

    # Format y-axis labels
    ax.set_yticklabels([f'{int(ds) // 1000}k' for ds in heatmap_r.index], rotation=0)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved heatmap to: {save_path}")

    plt.show()

    return fig


def create_top_results_scatter(corr_df, top_n=6,
                               save_path='../results/viz_top_results_scatter.png'):
    """
    Create scatter plots for top N results
    """

    # Load data
    data_df = pd.read_csv('../results/distribution_data_ALL.csv')

    # Get top N results for effective_dim_95
    metric_data = corr_df[corr_df['metric'] == 'effective_dim_95'].copy()
    metric_data = metric_data.sort_values('pearson_p').head(top_n)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(metric_data.iterrows()):
        if idx >= top_n:
            break

        ax = axes[idx]

        dict_size = row['dict_size']
        layer = row['layer']
        r = row['pearson_r']
        p = row['pearson_p']

        # Filter data for this combination
        subset = data_df[(data_df['dict_size'] == dict_size) &
                         (data_df['layer'] == layer)].copy()

        if len(subset) == 0:
            continue

        x = subset['effective_dim_95']
        y = subset['unlearning_difficulty']
        authors = subset['author']

        # Scatter plot
        ax.scatter(x, y, alpha=0.7, s=150, color='#2E86AB', edgecolors='black', linewidth=1.5)

        # Regression line
        z = np.polyfit(x, y, 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=2.5)

        # Annotate top 3 hardest authors
        top_hard = subset.nlargest(3, 'unlearning_difficulty')
        for _, author_row in top_hard.iterrows():
            ax.annotate(author_row['author'],
                        (author_row['effective_dim_95'], author_row['unlearning_difficulty']),
                        fontsize=8, alpha=0.8,
                        xytext=(5, 5), textcoords='offset points')

        # Labels
        ax.set_xlabel('Effective Dimensionality (95%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Unlearning Difficulty', fontsize=11, fontweight='bold')

        # Significance
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        title = f"#{idx + 1}: {dict_size // 1000}k SAE, Layer {layer}\nr={r:.3f}{sig}, p={p:.4f}, R²={row['r_squared']:.3f}"
        ax.set_title(title, fontsize=12, fontweight='bold')

        ax.grid(alpha=0.3, linestyle='--')

    plt.suptitle('Top 6 Correlations: Effective Dimensionality vs Unlearning Difficulty',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved scatter plots to: {save_path}")

    plt.show()

    return fig


def create_metric_comparison_plot(corr_df,
                                  save_path='../results/viz_metric_comparison.png'):
    """
    Compare different metrics (effective_dim, entropy, gini) across conditions
    """

    metrics = ['effective_dim_95', 'entropy', 'gini_coefficient']
    metric_labels = {
        'effective_dim_95': 'Effective Dim (95%)',
        'entropy': 'Entropy',
        'gini_coefficient': 'Gini Coefficient'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        metric_data = corr_df[corr_df['metric'] == metric].copy()

        # Create heatmap
        heatmap = metric_data.pivot_table(
            values='pearson_r',
            index='dict_size',
            columns='layer',
            aggfunc='first'
        )

        sns.heatmap(heatmap,
                    annot=False,
                    cmap='RdBu_r',
                    center=0,
                    vmin=-0.8,
                    vmax=0.8,
                    cbar_kws={'label': 'Pearson r'},
                    linewidths=0.5,
                    linecolor='gray',
                    ax=ax)

        ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dict Size', fontsize=12, fontweight='bold')
        ax.set_title(metric_labels[metric], fontsize=14, fontweight='bold')

        # Format y-axis
        ax.set_yticklabels([f'{int(ds) // 1000}k' for ds in heatmap.index], rotation=0)

    plt.suptitle('Distribution Metrics Comparison Across All Conditions',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved metric comparison to: {save_path}")

    plt.show()

    return fig


def create_layer_profile_plot(corr_df, metric='effective_dim_95',
                              save_path='../results/viz_layer_profile.png'):
    """
    Show how correlation changes across layers for each dict_size
    """

    metric_data = corr_df[corr_df['metric'] == metric].copy()

    fig, ax = plt.subplots(figsize=(12, 7))

    dict_sizes = sorted(metric_data['dict_size'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    for idx, dict_size in enumerate(dict_sizes):
        subset = metric_data[metric_data['dict_size'] == dict_size].copy()
        subset = subset.sort_values('layer')

        ax.plot(subset['layer'], subset['pearson_r'],
                marker=markers[idx], markersize=8, linewidth=2.5,
                label=f'{dict_size // 1000}k SAE', color=colors[idx], alpha=0.8)

        # Mark significant points
        significant = subset[subset['pearson_p'] < 0.05]
        if len(significant) > 0:
            ax.scatter(significant['layer'], significant['pearson_r'],
                       s=200, facecolors='none', edgecolors=colors[idx],
                       linewidths=3, alpha=0.8)

    # Horizontal line at r=0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Shaded region for strong correlation
    ax.axhspan(-0.5, -1.0, alpha=0.1, color='green', label='Strong Negative (r < -0.5)')
    ax.axhspan(0.5, 1.0, alpha=0.1, color='red', label='Strong Positive (r > 0.5)')

    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pearson Correlation (r)', fontsize=14, fontweight='bold')
    ax.set_title('Effective Dimensionality Correlation Across Layers\n(Circle = p < 0.05)',
                 fontsize=16, fontweight='bold')

    ax.set_xticks(range(0, 12))
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved layer profile to: {save_path}")

    plt.show()

    return fig


def create_dict_size_comparison(corr_df, metric='effective_dim_95',
                                save_path='../results/viz_dict_size_comparison.png'):
    """
    Compare dict_sizes: which is best overall?
    """

    metric_data = corr_df[corr_df['metric'] == metric].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Average |r| by dict_size
    ax = axes[0]

    dict_sizes = sorted(metric_data['dict_size'].unique())
    avg_abs_r = []

    for dict_size in dict_sizes:
        subset = metric_data[metric_data['dict_size'] == dict_size]
        avg_abs_r.append(abs(subset['pearson_r']).mean())

    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ax.bar([f'{ds // 1000}k' for ds in dict_sizes], avg_abs_r,
           color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_xlabel('SAE Dictionary Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average |Pearson r|', fontsize=14, fontweight='bold')
    ax.set_title('Average Correlation Strength by Dict Size', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Number of significant results by dict_size
    ax = axes[1]

    n_significant = []
    for dict_size in dict_sizes:
        subset = metric_data[metric_data['dict_size'] == dict_size]
        n_sig = (subset['pearson_p'] < 0.05).sum()
        n_significant.append(n_sig)

    ax.bar([f'{ds // 1000}k' for ds in dict_sizes], n_significant,
           color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_xlabel('SAE Dictionary Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Significant Results (p < 0.05)', fontsize=14, fontweight='bold')
    ax.set_title('Significant Results by Dict Size', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('SAE Dictionary Size Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved dict size comparison to: {save_path}")

    plt.show()

    return fig


def create_significance_map(corr_df, metric='effective_dim_95',
                            save_path='../results/viz_significance_map.png'):
    """
    Create a map showing which combinations are significant
    """

    metric_data = corr_df[corr_df['metric'] == metric].copy()

    # Create significance matrix
    sig_matrix = metric_data.pivot_table(
        values='pearson_p',
        index='dict_size',
        columns='layer',
        aggfunc='first'
    )

    # Convert to categorical: *** (p<0.001), ** (p<0.01), * (p<0.05), n.s.
    sig_categorical = sig_matrix.applymap(lambda p:
                                          3 if p < 0.001 else
                                          2 if p < 0.01 else
                                          1 if p < 0.05 else
                                          0 if pd.notna(p) else np.nan
                                          )

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create custom colormap
    from matplotlib.colors import ListedColormap
    colors_sig = ['#f0f0f0', '#90EE90', '#FFD700', '#FF6347']  # Gray, Light green, Gold, Red
    cmap = ListedColormap(colors_sig)

    sns.heatmap(sig_categorical,
                annot=False,
                cmap=cmap,
                cbar_kws={'label': 'Significance Level',
                          'ticks': [0, 1, 2, 3]},
                linewidths=0.5,
                linecolor='gray',
                ax=ax,
                vmin=0, vmax=3)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['n.s.', '* (p<0.05)', '** (p<0.01)', '*** (p<0.001)'])

    # Add correlation values as text
    heatmap_r = metric_data.pivot_table(
        values='pearson_r',
        index='dict_size',
        columns='layer',
        aggfunc='first'
    )

    for i, dict_size in enumerate(heatmap_r.index):
        for j, layer in enumerate(heatmap_r.columns):
            r = heatmap_r.iloc[i, j]
            if pd.notna(r):
                ax.text(j + 0.5, i + 0.5, f'{r:.2f}',
                        ha='center', va='center',
                        color='black', fontsize=9, fontweight='bold')

    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('SAE Dictionary Size', fontsize=14, fontweight='bold')
    ax.set_title('Statistical Significance Map: Effective Dimensionality vs Unlearning Difficulty',
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_yticklabels([f'{int(ds) // 1000}k' for ds in heatmap_r.index], rotation=0)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved significance map to: {save_path}")

    plt.show()

    return fig


def create_best_result_detailed(save_path='../results/viz_best_result_16k_layer8.png'):
    """
    Create detailed visualization for best result (16k Layer 8)
    """

    # Load data
    data_df = pd.read_csv('../results/distribution_data_ALL.csv')

    # Filter for best result
    best_data = data_df[(data_df['dict_size'] == 16384) & (data_df['layer'] == 8)].copy()

    if len(best_data) == 0:
        print("⚠️ No data found for 16k Layer 8")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Plot 1: Main scatter plot
    ax = axes[0, 0]

    x = best_data['effective_dim_95']
    y = best_data['unlearning_difficulty']

    ax.scatter(x, y, alpha=0.7, s=200, color='#2E86AB', edgecolors='black', linewidth=2)

    # Regression line
    z = np.polyfit(x, y, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=3)

    # Annotate ALL authors
    for _, row in best_data.iterrows():
        ax.annotate(row['author'],
                    (row['effective_dim_95'], row['unlearning_difficulty']),
                    fontsize=9, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')

    # Compute stats
    from scipy import stats as sp_stats
    r, p = sp_stats.pearsonr(x, y)

    ax.set_xlabel('Effective Dimensionality (95%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Unlearning Difficulty', fontsize=14, fontweight='bold')
    ax.set_title(f'Best Result: 16k SAE, Layer 8\nr={r:.3f}, p={p:.6f}, R²={r ** 2:.3f}',
                 fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    # Add stats box
    textstr = f'Pearson r = {r:.3f}***\nP-value = {p:.6f}\nR² = {r ** 2:.3f}\nN = {len(x)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # Plot 2: Entropy correlation
    ax = axes[0, 1]

    x2 = best_data['entropy']
    ax.scatter(x2, y, alpha=0.7, s=200, color='#A23B72', edgecolors='black', linewidth=2)

    z2 = np.polyfit(x2, y, 1)
    p2 = np.poly1d(z2)
    x2_line = np.linspace(x2.min(), x2.max(), 100)
    ax.plot(x2_line, p2(x2_line), "r--", alpha=0.8, linewidth=3)

    r2, p2_val = sp_stats.pearsonr(x2, y)

    ax.set_xlabel('Entropy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Unlearning Difficulty', fontsize=14, fontweight='bold')
    ax.set_title(f'Entropy at 16k Layer 8\nr={r2:.3f}, p={p2_val:.4f}',
                 fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    # Plot 3: Gini correlation
    ax = axes[1, 0]

    x3 = best_data['gini_coefficient']
    ax.scatter(x3, y, alpha=0.7, s=200, color='#F18F01', edgecolors='black', linewidth=2)

    z3 = np.polyfit(x3, y, 1)
    p3 = np.poly1d(z3)
    x3_line = np.linspace(x3.min(), x3.max(), 100)
    ax.plot(x3_line, p3(x3_line), "r--", alpha=0.8, linewidth=3)

    r3, p3_val = sp_stats.pearsonr(x3, y)

    ax.set_xlabel('Gini Coefficient', fontsize=14, fontweight='bold')
    ax.set_ylabel('Unlearning Difficulty', fontsize=14, fontweight='bold')
    ax.set_title(f'Gini Coefficient at 16k Layer 8\nr={r3:.3f}, p={p3_val:.4f}',
                 fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    # Plot 4: L0 correlation (for comparison)
    ax = axes[1, 1]

    x4 = best_data['l0_sparsity']
    ax.scatter(x4, y, alpha=0.7, s=200, color='#06A77D', edgecolors='black', linewidth=2)

    z4 = np.polyfit(x4, y, 1)
    p4 = np.poly1d(z4)
    x4_line = np.linspace(x4.min(), x4.max(), 100)
    ax.plot(x4_line, p4(x4_line), "r--", alpha=0.8, linewidth=3)

    r4, p4_val = sp_stats.pearsonr(x4, y)

    ax.set_xlabel('L0 Sparsity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Unlearning Difficulty', fontsize=14, fontweight='bold')
    ax.set_title(f'L0 Sparsity at 16k Layer 8 (for comparison)\nr={r4:.3f}, p={p4_val:.4f}',
                 fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    plt.suptitle('Detailed Analysis: Best Result (16k SAE, Layer 8)',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved detailed best result to: {save_path}")

    plt.show()

    return fig


def main():
    """
    Generate all visualizations
    """

    print("\n" + "=" * 80)
    print("COMPREHENSIVE VISUALIZATION GENERATION")
    print("=" * 80)

    # Load results
    print("\n1) Loading results...")
    corr_df = load_results()
    print(f"   ✓ Loaded {len(corr_df)} correlation results")

    # Create visualizations
    print("\n2) Creating visualizations...")

    print("\n   a) Heatmap for Effective Dimensionality...")
    create_heatmap_visualization(corr_df, metric='effective_dim_95')

    print("\n   b) Heatmap for Entropy...")
    create_heatmap_visualization(corr_df, metric='entropy',
                                 save_path='../results/viz_heatmap_entropy.png')

    print("\n   c) Heatmap for Gini Coefficient...")
    create_heatmap_visualization(corr_df, metric='gini_coefficient',
                                 save_path='../results/viz_heatmap_gini.png')

    print("\n   d) Top 6 results scatter plots...")
    create_top_results_scatter(corr_df)

    print("\n   e) Metric comparison...")
    create_metric_comparison_plot(corr_df)

    print("\n   f) Layer profile plot...")
    create_layer_profile_plot(corr_df)

    print("\n   g) Dict size comparison...")
    create_dict_size_comparison(corr_df)

    print("\n   h) Significance map...")
    create_significance_map(corr_df)

    print("\n   i) Best result detailed analysis...")
    create_best_result_detailed()

    # Summary
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print("\n📊 Generated visualizations:")
    print("   1. viz_heatmap_effective_dim_95.png")
    print("   2. viz_heatmap_entropy.png")
    print("   3. viz_heatmap_gini.png")
    print("   4. viz_top_results_scatter.png")
    print("   5. viz_metric_comparison.png")
    print("   6. viz_layer_profile.png")
    print("   7. viz_dict_size_comparison.png")
    print("   8. viz_significance_map.png")
    print("   9. viz_best_result_16k_layer8.png")
    print("\n✅ All visualizations complete!")


if __name__ == "__main__":
    main()