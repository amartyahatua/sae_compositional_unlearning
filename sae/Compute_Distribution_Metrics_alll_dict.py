import json
import numpy as np
from scipy import stats
from scipy.stats import entropy
import pandas as pd
from pathlib import Path


def load_superposition_data(filepath='../data/superposition_all_dict_sizes_COMPLETE.json'):
    """Load superposition measurement results"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def load_unlearning_results(filepath='../data/all_results_forget10.json'):
    """Load unlearning difficulty scores"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def compute_entropy(activations):
    """
    Compute entropy of activation distribution

    Higher entropy = more distributed
    Lower entropy = more concentrated
    """
    # Normalize to probability distribution
    probs = activations / (activations.sum() + 1e-10)

    # Compute Shannon entropy
    return entropy(probs)


def compute_participation_ratio(activations):
    """
    Compute participation ratio (from physics)

    Higher PR = more features participate equally
    Lower PR = few features dominate

    PR = (sum of activations)^2 / (sum of squared activations)
    """
    sum_acts = activations.sum()
    sum_squared_acts = (activations ** 2).sum()

    if sum_squared_acts == 0:
        return 0

    return (sum_acts ** 2) / sum_squared_acts


def compute_effective_dimensionality(activations, threshold=0.9):
    """
    Compute effective dimensionality

    Number of features needed to capture 'threshold' fraction of total activation

    Higher ED = need more features (distributed)
    Lower ED = need fewer features (concentrated)
    """
    sorted_acts = np.sort(activations)[::-1]  # Sort descending
    cumsum = np.cumsum(sorted_acts)
    total = cumsum[-1]

    if total == 0:
        return 0

    # Find where cumsum exceeds threshold * total
    n_effective = np.argmax(cumsum >= threshold * total) + 1

    return n_effective


def compute_gini_coefficient(activations):
    """
    Compute Gini coefficient (inequality measure)

    Higher Gini (→1) = more concentrated/unequal
    Lower Gini (→0) = more distributed/equal
    """
    sorted_acts = np.sort(activations)
    n = len(activations)
    cumsum = np.cumsum(sorted_acts)

    if cumsum[-1] == 0:
        return 0

    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def compute_top_k_concentration(activations, k=50):
    """
    Compute what fraction of total activation is in top-k features

    Higher = more concentrated in top features
    Lower = more distributed across all features
    """
    sorted_acts = np.sort(activations)[::-1]
    top_k_sum = sorted_acts[:k].sum()
    total = activations.sum()

    if total == 0:
        return 0

    return top_k_sum / total


def analyze_distribution_metrics(dict_size, layer_idx):
    """
    Compute all distribution metrics for a specific dict_size and layer
    """

    print(f"\n{'=' * 70}")
    print(f"DISTRIBUTION METRICS: Dict_size={dict_size}, Layer={layer_idx}")
    print(f"{'=' * 70}\n")

    # Load data
    superposition_data = load_superposition_data()
    unlearning_data = load_unlearning_results()

    sae_key = f'sae_{dict_size}'

    if sae_key not in superposition_data:
        print(f"❌ Dict_size {dict_size} not found in data")
        return None

    superposition_scores = superposition_data[sae_key]['superposition_scores']

    # Create unlearning difficulty map
    unlearn_map = {}
    for entry in unlearning_data:
        author = entry['author']
        unlearn_map[author] = entry.get('forget_increase', None)

    results = []

    for author in superposition_scores.keys():
        layer_key = str(layer_idx)

        if layer_key not in superposition_scores[author]:
            continue

        try:
            # Get activations
            author_acts = np.array(superposition_scores[author][layer_key]['author_mean_activation'])
            retain_acts = np.array(superposition_scores[author][layer_key]['retain_mean_activation'])

            # Get L0 sparsity
            threshold_key = 'threshold_0.01'
            l0_sparsity = superposition_scores[author][layer_key]['threshold_metrics'][threshold_key][
                'author_l0_sparsity']

            # Compute distribution metrics
            author_entropy = compute_entropy(author_acts)
            author_pr = compute_participation_ratio(author_acts)
            author_ed_90 = compute_effective_dimensionality(author_acts, threshold=0.9)
            author_ed_95 = compute_effective_dimensionality(author_acts, threshold=0.95)
            author_gini = compute_gini_coefficient(author_acts)
            author_top50_conc = compute_top_k_concentration(author_acts, k=50)
            author_top100_conc = compute_top_k_concentration(author_acts, k=100)

            # Also compute for retain (for comparison)
            retain_entropy = compute_entropy(retain_acts)
            retain_pr = compute_participation_ratio(retain_acts)

            # Get unlearning difficulty
            unlearn_diff = unlearn_map.get(author, None)

            results.append({
                'author': author,
                'dict_size': dict_size,
                'layer': layer_idx,
                'l0_sparsity': l0_sparsity,
                'entropy': author_entropy,
                'participation_ratio': author_pr,
                'effective_dim_90': author_ed_90,
                'effective_dim_95': author_ed_95,
                'gini_coefficient': author_gini,
                'top50_concentration': author_top50_conc,
                'top100_concentration': author_top100_conc,
                'retain_entropy': retain_entropy,
                'retain_participation_ratio': retain_pr,
                'unlearning_difficulty': unlearn_diff
            })

        except Exception as e:
            print(f"⚠️ Error processing {author}: {e}")
            continue

    df = pd.DataFrame(results)

    # Remove rows with missing difficulty
    df = df.dropna(subset=['unlearning_difficulty'])

    print(f"📊 Analyzed {len(df)} authors")

    return df


def compute_correlations(df):
    """
    Compute correlations between distribution metrics and unlearning difficulty
    """

    metrics = [
        'l0_sparsity',
        'entropy',
        'participation_ratio',
        'effective_dim_90',
        'effective_dim_95',
        'gini_coefficient',
        'top50_concentration',
        'top100_concentration'
    ]

    correlation_results = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Pearson correlation
        r, p = stats.pearsonr(df[metric], df['unlearning_difficulty'])

        # Spearman correlation
        rho, p_spearman = stats.spearmanr(df[metric], df['unlearning_difficulty'])

        correlation_results.append({
            'metric': metric,
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': p_spearman,
            'r_squared': r ** 2
        })

        # Print result
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {metric:25s}: r={r:+.3f}, p={p:.4f} {sig:3s} (R²={r ** 2:.3f})")

    return pd.DataFrame(correlation_results)


def comprehensive_analysis():
    """
    Run comprehensive analysis across all dict_sizes and layers
    """

    dict_sizes = [4096, 8192, 16384, 32768, 65536]
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # All layers

    # Storage for all results
    all_data = []
    all_correlations = []

    print("\n" + "=" * 80)
    print("COMPREHENSIVE DISTRIBUTION METRICS ANALYSIS")
    print("Testing ALL dict_sizes × ALL layers")
    print("=" * 80)

    # Test each combination
    for dict_size in dict_sizes:
        print(f"\n{'#' * 80}")
        print(f"DICT_SIZE: {dict_size}")
        print(f"{'#' * 80}")

        for layer in layers:
            try:
                # Analyze this combination
                df = analyze_distribution_metrics(dict_size=dict_size, layer_idx=layer)

                if df is None or len(df) == 0:
                    print(f"  ⚠️ No data for Layer {layer}")
                    continue

                # Compute correlations
                print(f"\n  Correlations for Layer {layer}:")
                corr_df = compute_correlations(df)

                # Add dict_size and layer to correlation results
                corr_df['dict_size'] = dict_size
                corr_df['layer'] = layer

                # Store results
                all_data.append(df)
                all_correlations.append(corr_df)

                # Save individual CSV for this dict_size-layer combination
                output_path = f'../results/distribution_data_dict{dict_size}_layer{layer}.csv'
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False, float_format='%.6f')

                # Find best metric
                best_idx = corr_df['pearson_p'].idxmin()
                best_metric = corr_df.loc[best_idx]
                sig = "***" if best_metric['pearson_p'] < 0.001 else "**" if best_metric['pearson_p'] < 0.01 else "*" if \
                best_metric['pearson_p'] < 0.05 else "n.s."

                print(
                    f"  ✓ Best: {best_metric['metric']} (r={best_metric['pearson_r']:.3f}, p={best_metric['pearson_p']:.4f} {sig})")

            except Exception as e:
                print(f"  ❌ Error at Layer {layer}: {e}")
                continue

    # Combine all results
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.to_csv('../results/distribution_data_ALL.csv', index=False, float_format='%.6f')
        print(f"\n💾 Saved combined data to: ../results/distribution_data_ALL.csv")

    if all_correlations:
        combined_corr = pd.concat(all_correlations, ignore_index=True)
        combined_corr.to_csv('../results/distribution_correlations_ALL.csv', index=False, float_format='%.6f')
        print(f"💾 Saved combined correlations to: ../results/distribution_correlations_ALL.csv")

    return combined_data, combined_corr


def create_summary_tables(combined_corr):
    """
    Create summary tables showing best results
    """

    print("\n" + "=" * 80)
    print("SUMMARY: BEST CORRELATIONS BY METRIC")
    print("=" * 80)

    metrics = ['effective_dim_95', 'entropy', 'gini_coefficient', 'l0_sparsity']

    for metric in metrics:
        metric_data = combined_corr[combined_corr['metric'] == metric].copy()

        if len(metric_data) == 0:
            continue

        # Sort by p-value (best first)
        metric_data = metric_data.sort_values('pearson_p')

        print(f"\n{'─' * 80}")
        print(f"METRIC: {metric.upper()}")
        print(f"{'─' * 80}")
        print(f"{'Rank':<6} {'Dict':<8} {'Layer':<7} {'r':<8} {'p-value':<10} {'R²':<8} {'Sig':<5}")
        print("─" * 80)

        for idx, (_, row) in enumerate(metric_data.head(10).iterrows()):
            sig = "***" if row['pearson_p'] < 0.001 else "**" if row['pearson_p'] < 0.01 else "*" if row[
                                                                                                         'pearson_p'] < 0.05 else "n.s."
            print(
                f"{idx + 1:<6} {row['dict_size'] // 1000:>2}k     {row['layer']:<7} {row['pearson_r']:+.3f}   {row['pearson_p']:.6f}   {row['r_squared']:.3f}   {sig:<5}")

    # Save summary
    summary_path = '../results/distribution_summary_best.csv'

    summary_rows = []
    for metric in metrics:
        metric_data = combined_corr[combined_corr['metric'] == metric].copy()
        if len(metric_data) > 0:
            best = metric_data.loc[metric_data['pearson_p'].idxmin()]
            summary_rows.append(best)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values('pearson_p')
        summary_df.to_csv(summary_path, index=False, float_format='%.6f')
        print(f"\n💾 Saved summary to: {summary_path}")


def create_heatmap_data(combined_corr, metric='effective_dim_95'):
    """
    Create heatmap data: dict_size × layer showing correlation strength
    """

    metric_data = combined_corr[combined_corr['metric'] == metric].copy()

    if len(metric_data) == 0:
        print(f"⚠️ No data for metric: {metric}")
        return

    # Pivot to create heatmap
    heatmap = metric_data.pivot_table(
        values='pearson_r',
        index='dict_size',
        columns='layer',
        aggfunc='first'
    )

    # Save
    heatmap_path = f'../results/heatmap_{metric}.csv'
    heatmap.to_csv(heatmap_path, float_format='%.3f')
    print(f"💾 Saved heatmap data for {metric} to: {heatmap_path}")

    # Print
    print(f"\n{'=' * 80}")
    print(f"HEATMAP: {metric.upper()} Correlation (Pearson r)")
    print(f"{'=' * 80}\n")
    print(heatmap.to_string())

    return heatmap


def main():
    """
    Main analysis pipeline
    """

    print("\n" + "=" * 80)
    print("COMPREHENSIVE DISTRIBUTION METRICS ANALYSIS")
    print("=" * 80)

    # Run comprehensive analysis
    print("\n1) Running analysis across ALL dict_sizes and layers...")
    combined_data, combined_corr = comprehensive_analysis()

    # Create summary tables
    print("\n2) Creating summary tables...")
    create_summary_tables(combined_corr)

    # Create heatmaps for key metrics
    print("\n3) Creating heatmap data...")
    for metric in ['effective_dim_95', 'entropy', 'gini_coefficient']:
        create_heatmap_data(combined_corr, metric=metric)

    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\n📊 Files generated:")
    print("  - distribution_data_ALL.csv (combined data)")
    print("  - distribution_correlations_ALL.csv (all correlations)")
    print("  - distribution_summary_best.csv (best results per metric)")
    print("  - heatmap_*.csv (heatmap data for visualization)")
    print("  - distribution_data_dict*_layer*.csv (individual combinations)")
    print("\n✅ All analyses complete!")


if __name__ == "__main__":
    main()