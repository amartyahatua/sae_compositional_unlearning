# # import json
# # import numpy as np
# # from scipy import stats
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from pathlib import Path
# #
# #
# # def load_superposition_data(filepath='../data/superposition_all_dict_sizes_COMPLETE.json'):
# #     """Load superposition measurement results"""
# #     with open(filepath, 'r') as f:
# #         data = json.load(f)
# #     return data
# #
# #
# # def load_unlearning_results(filepath='../data/all_results_jaccard_similarity.json'):
# #     """Load unlearning difficulty scores"""
# #     with open(filepath, 'r') as f:
# #         data = json.load(f)
# #     return data
# #
# #
# # def extract_correlation_data(superposition_data, unlearning_data, dict_size, layer_idx=11, threshold='0.0'):
# #     """
# #     Extract data for correlation analysis
# #
# #     Args:
# #         superposition_data: Full superposition results
# #         unlearning_data: Unlearning results
# #         dict_size: Which SAE dict_size to analyze
# #         layer_idx: Which layer to analyze (default 11)
# #         threshold: Which threshold to use (default '0.01')
# #
# #     Returns:
# #         authors, jaccard_scores, unlearning_difficulties
# #     """
# #
# #     sae_key = f'sae_{dict_size}'
# #
# #     if sae_key not in superposition_data:
# #         raise ValueError(f"Dict_size {dict_size} not found in superposition data")
# #
# #     authors = []
# #     jaccard_scores = []
# #     unlearning_difficulties = []
# #
# #     superposition_scores = superposition_data[sae_key]['superposition_scores']
# #
# #     for author in superposition_scores.keys():
# #         # Get superposition score for this author at specified layer
# #         layer_key = str(layer_idx)
# #
# #         if layer_key not in superposition_scores[author]:
# #             print(f"Warning: Layer {layer_idx} not found for {author}")
# #             continue
# #
# #         # Extract Jaccard similarity at specified threshold
# #         threshold_key = f'threshold_{threshold}'
# #         if threshold_key in superposition_scores[author][layer_key]['threshold_metrics']:
# #             jaccard = superposition_scores[author][layer_key]['threshold_metrics'][threshold_key]['jaccard_similarity']
# #         else:
# #             print(f"Warning: Threshold {threshold} not found for {author}")
# #             continue
# #
# #         # Get unlearning difficulty for this author
# #         for count in range(len(unlearning_data)):
# #             if author in unlearning_data[count]['author']:
# #                 # Extract forget_increase metric (or whatever metric you use)
# #                 unlearn_diff = unlearning_data[count].get('forget_increase', None)
# #
# #                 if unlearn_diff is None:
# #                     print(f"Warning: No unlearning difficulty for {author}")
# #                     continue
# #
# #                 authors.append(author)
# #                 jaccard_scores.append(jaccard)
# #                 unlearning_difficulties.append(unlearn_diff)
# #             # else:
# #             #     print(f"Warning: {author} not found in unlearning data")
# #
# #     return authors, np.array(jaccard_scores), np.array(unlearning_difficulties)
# #
# #
# # def compute_correlation_statistics(jaccard_scores, unlearning_difficulties):
# #     """
# #     Compute correlation statistics
# #
# #     Returns:
# #         dict with Pearson r, Spearman rho, p-values, R^2
# #     """
# #
# #     # Pearson correlation (linear)
# #     pearson_r, pearson_p = stats.pearsonr(jaccard_scores, unlearning_difficulties)
# #
# #     # Spearman correlation (rank-based, robust to outliers)
# #     spearman_rho, spearman_p = stats.spearmanr(jaccard_scores, unlearning_difficulties)
# #
# #     # R-squared (variance explained)
# #     r_squared = pearson_r ** 2
# #
# #     return {
# #         'pearson_r': pearson_r,
# #         'pearson_p': pearson_p,
# #         'spearman_rho': spearman_rho,
# #         'spearman_p': spearman_p,
# #         'r_squared': r_squared,
# #         'n_samples': len(jaccard_scores)
# #     }
# #
# #
# # def analyze_all_dict_sizes(superposition_data, unlearning_data, layer_idx=11, threshold='0.01'):
# #     """
# #     Analyze correlation for all dictionary sizes
# #
# #     Returns:
# #         dict with results for each dict_size
# #     """
# #
# #     dict_sizes = [4096, 8192, 16384, 32768, 65536]
# #
# #     results = {}
# #
# #     print(f"\n{'=' * 70}")
# #     print(f"CORRELATION ANALYSIS: Layer {layer_idx}, Threshold {threshold}")
# #     print(f"{'=' * 70}\n")
# #
# #     for dict_size in dict_sizes:
# #         print(f"\nDict_size: {dict_size}")
# #         print("-" * 40)
# #
# #         try:
# #             authors, jaccard, unlearn_diff = extract_correlation_data(
# #                 superposition_data, unlearning_data, dict_size, layer_idx, threshold
# #             )
# #
# #             stats_result = compute_correlation_statistics(jaccard, unlearn_diff)
# #
# #             results[dict_size] = {
# #                 'authors': authors,
# #                 'jaccard_scores': jaccard.tolist(),
# #                 'unlearning_difficulties': unlearn_diff.tolist(),
# #                 'statistics': stats_result
# #             }
# #
# #             # Print results
# #             print(f"  N = {stats_result['n_samples']} authors")
# #             print(f"  Pearson r   = {stats_result['pearson_r']:+.3f}, p = {stats_result['pearson_p']:.4f}")
# #             print(f"  Spearman ρ  = {stats_result['spearman_rho']:+.3f}, p = {stats_result['spearman_p']:.4f}")
# #             print(f"  R² = {stats_result['r_squared']:.3f} ({100 * stats_result['r_squared']:.1f}% variance explained)")
# #
# #             # Significance stars
# #             if stats_result['pearson_p'] < 0.001:
# #                 sig = "***"
# #             elif stats_result['pearson_p'] < 0.01:
# #                 sig = "**"
# #             elif stats_result['pearson_p'] < 0.05:
# #                 sig = "*"
# #             else:
# #                 sig = "n.s."
# #             print(f"  Significance: {sig}")
# #
# #         except Exception as e:
# #             print(f"  ERROR: {e}")
# #             results[dict_size] = None
# #
# #     return results
# #
# #
# # def plot_correlation_trends(results, save_path='../results/correlation_trends.png'):
# #     """
# #     Plot how correlation strength changes with dict_size
# #     """
# #
# #     dict_sizes = []
# #     pearson_rs = []
# #     pearson_ps = []
# #     r_squareds = []
# #
# #     for dict_size, data in sorted(results.items()):
# #         if data is not None:
# #             dict_sizes.append(dict_size)
# #             pearson_rs.append(abs(data['statistics']['pearson_r']))  # Absolute value
# #             pearson_ps.append(data['statistics']['pearson_p'])
# #             r_squareds.append(data['statistics']['r_squared'])
# #
# #     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# #
# #     # Plot 1: Correlation strength vs dict_size
# #     ax = axes[0]
# #     ax.plot(dict_sizes, pearson_rs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
# #     ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Moderate correlation')
# #     ax.set_xlabel('Dictionary Size', fontsize=12)
# #     ax.set_ylabel('|Pearson r|', fontsize=12)
# #     ax.set_title('Correlation Strength vs Dict Size', fontsize=13, fontweight='bold')
# #     ax.set_xscale('log')
# #     ax.set_xticks(dict_sizes)
# #     ax.set_xticklabels([f'{d // 1000}k' for d in dict_sizes])
# #     ax.grid(alpha=0.3)
# #     ax.legend()
# #
# #     # Plot 2: P-value vs dict_size
# #     ax = axes[1]
# #     ax.plot(dict_sizes, pearson_ps, 'o-', linewidth=2, markersize=8, color='#A23B72')
# #     ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='p = 0.05')
# #     ax.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.5, label='p = 0.01')
# #     ax.set_xlabel('Dictionary Size', fontsize=12)
# #     ax.set_ylabel('P-value', fontsize=12)
# #     ax.set_title('Statistical Significance vs Dict Size', fontsize=13, fontweight='bold')
# #     ax.set_xscale('log')
# #     ax.set_yscale('log')
# #     ax.set_xticks(dict_sizes)
# #     ax.set_xticklabels([f'{d // 1000}k' for d in dict_sizes])
# #     ax.grid(alpha=0.3)
# #     ax.legend()
# #
# #     # Plot 3: R² vs dict_size
# #     ax = axes[2]
# #     ax.plot(dict_sizes, r_squareds, 'o-', linewidth=2, markersize=8, color='#F18F01')
# #     ax.set_xlabel('Dictionary Size', fontsize=12)
# #     ax.set_ylabel('R² (Variance Explained)', fontsize=12)
# #     ax.set_title('Predictive Power vs Dict Size', fontsize=13, fontweight='bold')
# #     ax.set_xscale('log')
# #     ax.set_xticks(dict_sizes)
# #     ax.set_xticklabels([f'{d // 1000}k' for d in dict_sizes])
# #     ax.grid(alpha=0.3)
# #
# #     plt.tight_layout()
# #
# #     # Create directory if needed
# #     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
# #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #     print(f"\n📊 Saved trend plot to: {save_path}")
# #
# #     plt.show()
# #
# #
# # def plot_scatter_all_dict_sizes(results, save_path='../results/correlation_scatter_all.png'):
# #     """
# #     Plot scatter plots for all dict_sizes
# #     """
# #
# #     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# #     axes = axes.flatten()
# #
# #     for idx, (dict_size, data) in enumerate(sorted(results.items())):
# #         if data is None:
# #             continue
# #
# #         ax = axes[idx]
# #
# #         jaccard = np.array(data['jaccard_scores'])
# #         unlearn = np.array(data['unlearning_difficulties'])
# #         authors = data['authors']
# #         stats_data = data['statistics']
# #
# #         # Scatter plot
# #         ax.scatter(jaccard, unlearn, alpha=0.6, s=100, color='#2E86AB')
# #
# #         # Add regression line
# #         z = np.polyfit(jaccard, unlearn, 1)
# #         p = np.poly1d(z)
# #         x_line = np.linspace(jaccard.min(), jaccard.max(), 100)
# #         ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
# #
# #         # Annotate outliers (top 3 highest unlearning difficulty)
# #         top_indices = np.argsort(unlearn)[-3:]
# #         for i in top_indices:
# #             ax.annotate(authors[i], (jaccard[i], unlearn[i]),
# #                         fontsize=8, alpha=0.7,
# #                         xytext=(5, 5), textcoords='offset points')
# #
# #         # Labels and title
# #         ax.set_xlabel('Jaccard Similarity', fontsize=11)
# #         ax.set_ylabel('Unlearning Difficulty', fontsize=11)
# #
# #         sig = "***" if stats_data['pearson_p'] < 0.001 else "**" if stats_data['pearson_p'] < 0.01 else "*" if \
# #         stats_data['pearson_p'] < 0.05 else "n.s."
# #         title = f"Dict={dict_size // 1000}k: r={stats_data['pearson_r']:.2f}{sig}, R²={stats_data['r_squared']:.2f}"
# #         ax.set_title(title, fontsize=12, fontweight='bold')
# #         ax.grid(alpha=0.3)
# #
# #     # Hide unused subplot
# #     if len(results) < 6:
# #         axes[5].axis('off')
# #
# #     plt.tight_layout()
# #
# #     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
# #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #     print(f"📊 Saved scatter plots to: {save_path}")
# #
# #     plt.show()
# #
# #
# # def save_correlation_results(results, save_path='../data/correlation_analysis_results_sae_all_dict.json'):
# #     """Save correlation analysis results to JSON"""
# #
# #     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
# #
# #     with open(save_path, 'w') as f:
# #         json.dump(results, f, indent=2)
# #
# #     print(f"\n💾 Saved correlation results to: {save_path}")
# #
# #
# # def main():
# #     """Main analysis pipeline"""
# #
# #     print("\n" + "=" * 70)
# #     print("PHASE 2: CORRELATION ANALYSIS")
# #     print("=" * 70)
# #
# #     # Load data
# #     print("\n1) Loading data...")
# #     superposition_data = load_superposition_data()
# #     unlearning_data = load_unlearning_results()
# #
# #     print(f"   ✓ Loaded superposition data for {len(superposition_data)} dict_sizes")
# #     print(f"   ✓ Loaded unlearning data for {len(unlearning_data)} authors")
# #
# #     # Analyze correlations
# #     print("\n2) Computing correlations...")
# #
# #     for layer in range(12):
# #         results = analyze_all_dict_sizes(
# #             superposition_data,
# #             unlearning_data,
# #             layer_idx=layer,  # Focus on Layer 11
# #             threshold='0.0'  # Use threshold 0.01
# #         )
# #
# #         # Save results
# #         print("\n3) Saving results...")
# #         save_correlation_results(results)
# #
# #         # Create visualizations
# #         # print("\n4) Creating visualizations...")
# #         # plot_correlation_trends(results)
# #         # plot_scatter_all_dict_sizes(results)
# #
# #         # Summary
# #         print("\n" + "=" * 70)
# #         print("SUMMARY")
# #         print("=" * 70)
# #
# #         for dict_size in sorted(results.keys()):
# #             if results[dict_size] is not None:
# #                 stats_data = results[dict_size]['statistics']
# #                 sig = "✓" if stats_data['pearson_p'] < 0.05 else "✗"
# #                 print(f"{sig} {dict_size // 1000:2d}k: r={stats_data['pearson_r']:+.3f}, p={stats_data['pearson_p']:.4f}")
# #
# #         print("\n✅ Correlation analysis complete!")
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
# import json
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
#
#
# def load_superposition_data(filepath='../data/superposition_all_dict_sizes_COMPLETE.json'):
#     """Load superposition measurement results"""
#     with open(filepath, 'r') as f:
#         data = json.load(f)
#     return data
#
#
# def load_unlearning_results(filepath='../data/all_results_forget10.json'):
#     """Load unlearning difficulty scores"""
#     with open(filepath, 'r') as f:
#         data = json.load(f)
#     return data
#
#
# def extract_correlation_data(superposition_data, unlearning_data, dict_size, layer_idx=11, threshold='0.01',
#                              metric='jaccard'):
#     """
#     Extract data for correlation analysis
#
#     Args:
#         superposition_data: Full superposition results
#         unlearning_data: Unlearning results
#         dict_size: Which SAE dict_size to analyze
#         layer_idx: Which layer to analyze (default 11)
#         threshold: Which threshold to use (default '0.01')
#         metric: 'jaccard', 'cosine', 'l2', or 'l0'
#
#     Returns:
#         authors, metric_scores, unlearning_difficulties
#     """
#
#     sae_key = f'sae_{dict_size}'
#
#     if sae_key not in superposition_data:
#         raise ValueError(f"Dict_size {dict_size} not found in superposition data")
#
#     authors = []
#     metric_scores = []
#     unlearning_difficulties = []
#
#     superposition_scores = superposition_data[sae_key]['superposition_scores']
#
#     for author in superposition_scores.keys():
#         # Get superposition score for this author at specified layer
#         layer_key = str(layer_idx)
#
#         if layer_key not in superposition_scores[author]:
#             print(f"Warning: Layer {layer_idx} not found for {author}")
#             continue
#
#         # Extract the requested metric
#         try:
#             if metric == 'jaccard':
#                 threshold_key = f'threshold_{threshold}'
#                 score = superposition_scores[author][layer_key]['threshold_metrics'][threshold_key][
#                     'jaccard_similarity']
#             elif metric == 'cosine':
#                 score = superposition_scores[author][layer_key]['cosine_similarity']
#             elif metric == 'l2':
#                 score = superposition_scores[author][layer_key]['l2_distance']
#             elif metric == 'l0':
#                 threshold_key = f'threshold_{threshold}'
#                 score = superposition_scores[author][layer_key]['threshold_metrics'][threshold_key][
#                     'author_l0_sparsity']
#             else:
#                 raise ValueError(f"Unknown metric: {metric}")
#         except KeyError as e:
#             print(f"Warning: Metric {metric} not found for {author}: {e}")
#             continue
#
#         # Get unlearning difficulty for this author
#         for count in range(len(unlearning_data)):
#             if author in unlearning_data[count]['author']:
#                 unlearn_diff = unlearning_data[count].get('forget_increase', None)
#
#                 if unlearn_diff is None:
#                     print(f"Warning: No unlearning difficulty for {author}")
#                     continue
#
#                 authors.append(author)
#                 metric_scores.append(score)
#                 unlearning_difficulties.append(unlearn_diff)
#             # else:
#             #     print(f"Warning: {author} not found in unlearning data")
#
#     return authors, np.array(metric_scores), np.array(unlearning_difficulties)
#
#
# def compute_correlation_statistics(metric_scores, unlearning_difficulties):
#     """
#     Compute correlation statistics
#
#     Returns:
#         dict with Pearson r, Spearman rho, p-values, R^2
#     """
#
#     # Pearson correlation (linear)
#     pearson_r, pearson_p = stats.pearsonr(metric_scores, unlearning_difficulties)
#
#     # Spearman correlation (rank-based, robust to outliers)
#     spearman_rho, spearman_p = stats.spearmanr(metric_scores, unlearning_difficulties)
#
#     # R-squared (variance explained)
#     r_squared = pearson_r ** 2
#
#     return {
#         'pearson_r': pearson_r,
#         'pearson_p': pearson_p,
#         'spearman_rho': spearman_rho,
#         'spearman_p': spearman_p,
#         'r_squared': r_squared,
#         'n_samples': len(metric_scores)
#     }
#
#
# def analyze_all_dict_sizes(superposition_data, unlearning_data, layer_idx=11, threshold='0.01', metric='jaccard'):
#     """
#     Analyze correlation for all dictionary sizes
#
#     Args:
#         metric: 'jaccard', 'cosine', 'l2', or 'l0'
#
#     Returns:
#         dict with results for each dict_size
#     """
#
#     dict_sizes = [4096, 8192, 16384, 32768, 65536]
#
#     results = {}
#
#     print(f"\n{'=' * 70}")
#     print(f"CORRELATION ANALYSIS: Metric={metric}, Layer={layer_idx}, Threshold={threshold}")
#     print(f"{'=' * 70}\n")
#
#     for dict_size in dict_sizes:
#         print(f"\nDict_size: {dict_size}")
#         print("-" * 40)
#
#         try:
#             authors, metric_scores, unlearn_diff = extract_correlation_data(
#                 superposition_data, unlearning_data, dict_size, layer_idx, threshold, metric
#             )
#
#             stats_result = compute_correlation_statistics(metric_scores, unlearn_diff)
#
#             results[dict_size] = {
#                 'authors': authors,
#                 'metric_scores': metric_scores.tolist(),
#                 'unlearning_difficulties': unlearn_diff.tolist(),
#                 'statistics': stats_result
#             }
#
#             # Print results
#             print(f"  N = {stats_result['n_samples']} authors")
#             print(f"  Pearson r   = {stats_result['pearson_r']:+.3f}, p = {stats_result['pearson_p']:.4f}")
#             print(f"  Spearman ρ  = {stats_result['spearman_rho']:+.3f}, p = {stats_result['spearman_p']:.4f}")
#             print(f"  R² = {stats_result['r_squared']:.3f} ({100 * stats_result['r_squared']:.1f}% variance explained)")
#
#             # Significance stars
#             if stats_result['pearson_p'] < 0.001:
#                 sig = "***"
#             elif stats_result['pearson_p'] < 0.01:
#                 sig = "**"
#             elif stats_result['pearson_p'] < 0.05:
#                 sig = "*"
#             else:
#                 sig = "n.s."
#             print(f"  Significance: {sig}")
#
#         except Exception as e:
#             print(f"  ERROR: {e}")
#             results[dict_size] = None
#
#     return results
#
#
# def plot_correlation_trends(all_results, save_path='../results/correlation_trends_all_metrics.png'):
#     """
#     Plot how correlation strength changes with dict_size for all metrics
#     """
#
#     metrics = list(all_results.keys())
#     dict_sizes = [4096, 8192, 16384, 32768, 65536]
#
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#     axes = axes.flatten()
#
#     colors = {'jaccard': '#2E86AB', 'cosine': '#A23B72', 'l2': '#F18F01', 'l0': '#06A77D'}
#
#     for metric_idx, metric in enumerate(metrics):
#         ax = axes[metric_idx]
#         results = all_results[metric]
#
#         pearson_rs = []
#         pearson_ps = []
#
#         for dict_size in dict_sizes:
#             if results.get(dict_size) and results[dict_size] is not None:
#                 pearson_rs.append(abs(results[dict_size]['statistics']['pearson_r']))
#                 pearson_ps.append(results[dict_size]['statistics']['pearson_p'])
#             else:
#                 pearson_rs.append(np.nan)
#                 pearson_ps.append(np.nan)
#
#         # Plot correlation strength
#         ax.plot(dict_sizes, pearson_rs, 'o-', linewidth=2, markersize=8,
#                 color=colors.get(metric, 'gray'), label='|Pearson r|')
#         ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Moderate (0.5)')
#         ax.axhline(y=0.7, color='darkred', linestyle='--', alpha=0.5, label='Strong (0.7)')
#
#         # Mark significant points
#         for i, (ds, r, p) in enumerate(zip(dict_sizes, pearson_rs, pearson_ps)):
#             if p < 0.05:
#                 ax.scatter([ds], [r], s=200, facecolors='none', edgecolors='red', linewidths=3)
#
#         ax.set_xlabel('Dictionary Size', fontsize=12)
#         ax.set_ylabel('|Pearson r|', fontsize=12)
#         ax.set_title(f'Metric: {metric.upper()}', fontsize=13, fontweight='bold')
#         ax.set_xscale('log')
#         ax.set_xticks(dict_sizes)
#         ax.set_xticklabels([f'{d // 1000}k' for d in dict_sizes])
#         ax.grid(alpha=0.3)
#         ax.legend()
#         ax.set_ylim([0, 1])
#
#     plt.tight_layout()
#
#     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"\n📊 Saved trend plot to: {save_path}")
#
#     plt.show()
#
#
# def plot_scatter_best_metric(all_results, save_path='../results/correlation_scatter_best.png'):
#     """
#     Plot scatter plots for the metric that shows the strongest correlation
#     """
#
#     # Find best metric (highest average |r| across dict_sizes)
#     best_metric = None
#     best_avg_r = 0
#
#     for metric, results in all_results.items():
#         avg_r = np.nanmean([abs(results[ds]['statistics']['pearson_r'])
#                             for ds in results.keys() if results[ds] is not None])
#         if avg_r > best_avg_r:
#             best_avg_r = avg_r
#             best_metric = metric
#
#     print(f"\n🏆 Best metric: {best_metric} (avg |r| = {best_avg_r:.3f})")
#
#     results = all_results[best_metric]
#     dict_sizes = [k for k in sorted(results.keys()) if results[k] is not None]
#
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.flatten()
#
#     for idx, dict_size in enumerate(dict_sizes):
#         if idx >= 6:
#             break
#
#         data = results[dict_size]
#         ax = axes[idx]
#
#         metric_scores = np.array(data['metric_scores'])
#         unlearn = np.array(data['unlearning_difficulties'])
#         authors = data['authors']
#         stats_data = data['statistics']
#
#         # Scatter plot
#         ax.scatter(metric_scores, unlearn, alpha=0.6, s=100, color='#2E86AB')
#
#         # Add regression line
#         z = np.polyfit(metric_scores, unlearn, 1)
#         p = np.poly1d(z)
#         x_line = np.linspace(metric_scores.min(), metric_scores.max(), 100)
#         ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
#
#         # Annotate outliers (top 3 highest unlearning difficulty)
#         top_indices = np.argsort(unlearn)[-3:]
#         for i in top_indices:
#             ax.annotate(authors[i], (metric_scores[i], unlearn[i]),
#                         fontsize=8, alpha=0.7,
#                         xytext=(5, 5), textcoords='offset points')
#
#         # Labels and title
#         ax.set_xlabel(f'{best_metric.upper()} Score', fontsize=11)
#         ax.set_ylabel('Unlearning Difficulty', fontsize=11)
#
#         sig = "***" if stats_data['pearson_p'] < 0.001 else "**" if stats_data['pearson_p'] < 0.01 else "*" if \
#         stats_data['pearson_p'] < 0.05 else "n.s."
#         title = f"Dict={dict_size // 1000}k: r={stats_data['pearson_r']:.2f}{sig}, R²={stats_data['r_squared']:.2f}"
#         ax.set_title(title, fontsize=12, fontweight='bold')
#         ax.grid(alpha=0.3)
#
#     # Hide unused subplots
#     for idx in range(len(dict_sizes), 6):
#         axes[idx].axis('off')
#
#     plt.suptitle(f'Best Metric: {best_metric.upper()}', fontsize=14, fontweight='bold')
#     plt.tight_layout()
#
#     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"📊 Saved scatter plots to: {save_path}")
#
#     plt.show()
#
#
# def save_correlation_results(all_results, save_path='../results/correlation_analysis_all_metrics.json'):
#     """Save correlation analysis results to JSON"""
#
#     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#
#     with open(save_path, 'w') as f:
#         json.dump(all_results, f, indent=2)
#
#     print(f"\n💾 Saved correlation results to: {save_path}")
#
#
# def print_summary_table(all_results):
#     """Print a summary table of all metrics and dict_sizes"""
#
#     print("\n" + "=" * 90)
#     print("SUMMARY TABLE: Pearson r (p-value)")
#     print("=" * 90)
#
#     dict_sizes = [4096, 8192, 16384, 32768, 65536]
#     metrics = list(all_results.keys())
#
#     # Header
#     header = f"{'Metric':<10}"
#     for ds in dict_sizes:
#         header += f"{ds // 1000:>8}k"
#     print(header)
#     print("-" * 90)
#
#     # Rows
#     for metric in metrics:
#         row = f"{metric:<10}"
#         for dict_size in dict_sizes:
#             if all_results[metric].get(dict_size) and all_results[metric][dict_size] is not None:
#                 r = all_results[metric][dict_size]['statistics']['pearson_r']
#                 p = all_results[metric][dict_size]['statistics']['pearson_p']
#
#                 # Color code by significance
#                 if p < 0.001:
#                     sig = "***"
#                 elif p < 0.01:
#                     sig = "**"
#                 elif p < 0.05:
#                     sig = "*"
#                 else:
#                     sig = ""
#
#                 row += f"{r:+.2f}{sig:<3}"
#             else:
#                 row += "     N/A"
#         print(row)
#
#     print("=" * 90)
#     print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
#
#
# def main():
#     """Main analysis pipeline"""
#
#     print("\n" + "=" * 70)
#     print("PHASE 2: CORRELATION ANALYSIS - ALL METRICS")
#     print("=" * 70)
#
#     # Load data
#     print("\n1) Loading data...")
#     superposition_data = load_superposition_data()
#     unlearning_data = load_unlearning_results()
#
#     print(f"   ✓ Loaded superposition data for {len(superposition_data)} dict_sizes")
#     print(f"   ✓ Loaded unlearning data for {len(unlearning_data)} authors")
#
#     # Test all metrics
#     metrics_to_test = ['jaccard', 'cosine', 'l2', 'l0']
#     all_results = {}
#
#     print("\n2) Computing correlations for all metrics...")
#     for layer in range(12):
#         for thrsld in ['0.0', '0.01', '0.001', '0.1']:
#             for metric in metrics_to_test:
#                 print(f"\n{'#' * 70}")
#                 print(f"TESTING METRIC: {metric.upper()}")
#                 print(f"{'#' * 70}")
#
#                 results = analyze_all_dict_sizes(
#                     superposition_data,
#                     unlearning_data,
#                     layer_idx=layer,
#                     threshold=thrsld,
#                     metric=metric
#                 )
#
#                 all_results[metric] = results
#
#     # Save results
#     print("\n3) Saving results...")
#     save_correlation_results(all_results)
#
#     # Print summary table
#     print("\n4) Summary across all metrics...")
#     print_summary_table(all_results)
#
#     # Create visualizations
#     # print("\n5) Creating visualizations...")
#     # plot_correlation_trends(all_results)
#     # plot_scatter_best_metric(all_results)
#
#     print("\n✅ Correlation analysis complete for all metrics!")
#
#
# if __name__ == "__main__":
#     main()

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


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


def extract_correlation_data(superposition_data, unlearning_data, dict_size, layer_idx, threshold='0.01',
                             metric='jaccard'):
    """
    Extract data for correlation analysis

    Args:
        superposition_data: Full superposition results
        unlearning_data: Unlearning results
        dict_size: Which SAE dict_size to analyze
        layer_idx: Which layer to analyze (default 11)
        threshold: Which threshold to use (default '0.01')
        metric: 'jaccard', 'cosine', 'l2', or 'l0'

    Returns:
        authors, metric_scores, unlearning_difficulties
    """

    sae_key = f'sae_{dict_size}'

    if sae_key not in superposition_data:
        raise ValueError(f"Dict_size {dict_size} not found in superposition data")

    authors = []
    metric_scores = []
    unlearning_difficulties = []

    superposition_scores = superposition_data[sae_key]['superposition_scores']

    for author in superposition_scores.keys():
        # Get superposition score for this author at specified layer
        layer_key = str(layer_idx)

        if layer_key not in superposition_scores[author]:
            print(f"Warning: Layer {layer_idx} not found for {author}")
            continue

        # Extract the requested metric
        try:
            if metric == 'jaccard':
                threshold_key = f'threshold_{threshold}'
                score = superposition_scores[author][layer_key]['threshold_metrics'][threshold_key][
                    'jaccard_similarity']
            elif metric == 'cosine':
                score = superposition_scores[author][layer_key]['cosine_similarity']
            elif metric == 'l2':
                score = superposition_scores[author][layer_key]['l2_distance']
            elif metric == 'l0':
                threshold_key = f'threshold_{threshold}'
                score = superposition_scores[author][layer_key]['threshold_metrics'][threshold_key][
                    'author_l0_sparsity']
            else:
                raise ValueError(f"Unknown metric: {metric}")
        except KeyError as e:
            print(f"Warning: Metric {metric} not found for {author}: {e}")
            continue

        # Get unlearning difficulty for this author
        for count in range(len(unlearning_data)):
            if author in unlearning_data[count]['author']:
                unlearn_diff = unlearning_data[count].get('forget_increase', None)

                if unlearn_diff is None:
                    print(f"Warning: No unlearning difficulty for {author}")
                    continue

                authors.append(author)
                metric_scores.append(score)
                unlearning_difficulties.append(unlearn_diff)

    return authors, np.array(metric_scores), np.array(unlearning_difficulties)


def compute_correlation_statistics(metric_scores, unlearning_difficulties):
    """
    Compute correlation statistics

    Returns:
        dict with Pearson r, Spearman rho, p-values, R^2
    """

    # Pearson correlation (linear)
    pearson_r, pearson_p = stats.pearsonr(metric_scores, unlearning_difficulties)

    # Spearman correlation (rank-based, robust to outliers)
    spearman_rho, spearman_p = stats.spearmanr(metric_scores, unlearning_difficulties)

    # R-squared (variance explained)
    r_squared = pearson_r ** 2

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'r_squared': r_squared,
        'n_samples': len(metric_scores)
    }


def analyze_all_dict_sizes(superposition_data, unlearning_data, layer_idx=11, threshold='0.01', metric='jaccard'):
    """
    Analyze correlation for all dictionary sizes

    Args:
        metric: 'jaccard', 'cosine', 'l2', or 'l0'

    Returns:
        dict with results for each dict_size
    """

    dict_sizes = [4096, 8192, 16384, 32768, 65536]

    results = {}

    print(f"\n{'=' * 70}")
    print(f"CORRELATION ANALYSIS: Metric={metric}, Layer={layer_idx}, Threshold={threshold}")
    print(f"{'=' * 70}\n")

    for dict_size in dict_sizes:
        print(f"\nDict_size: {dict_size}")
        print("-" * 40)

        try:
            authors, metric_scores, unlearn_diff = extract_correlation_data(
                superposition_data, unlearning_data, dict_size, layer_idx, threshold, metric
            )

            stats_result = compute_correlation_statistics(metric_scores, unlearn_diff)

            results[dict_size] = {
                'authors': authors,
                'metric_scores': metric_scores.tolist(),
                'unlearning_difficulties': unlearn_diff.tolist(),
                'statistics': stats_result
            }

            # Print results
            print(f"  N = {stats_result['n_samples']} authors")
            print(f"  Pearson r   = {stats_result['pearson_r']:+.3f}, p = {stats_result['pearson_p']:.4f}")
            print(f"  Spearman ρ  = {stats_result['spearman_rho']:+.3f}, p = {stats_result['spearman_p']:.4f}")
            print(f"  R² = {stats_result['r_squared']:.3f} ({100 * stats_result['r_squared']:.1f}% variance explained)")

            # Significance stars
            if stats_result['pearson_p'] < 0.001:
                sig = "***"
            elif stats_result['pearson_p'] < 0.01:
                sig = "**"
            elif stats_result['pearson_p'] < 0.05:
                sig = "*"
            else:
                sig = "n.s."
            print(f"  Significance: {sig}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[dict_size] = None

    return results


def save_correlation_results(all_results, layer):
    """Save correlation analysis results to JSON"""
    save_path = f'../data/diff_dict_size/correlation_analysis_all_metrics_layer_{layer}.json'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Saved correlation results (JSON) to: {save_path}")


def save_correlation_results_csv(all_results, layer):
    """Save correlation analysis results to CSV format"""

    save_path = f'../data/diff_dict_size/correlation_analysis_all_metrics_layer_{layer}.csv'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Create a list to store all rows
    rows = []

    for metric, results in all_results.items():
        for dict_size, data in sorted(results.items()):
            if data is not None:
                stats = data['statistics']

                # Create a row for this metric-dict_size combination
                row = {
                    'metric': metric,
                    'dict_size': dict_size,
                    'n_samples': stats['n_samples'],
                    'pearson_r': stats['pearson_r'],
                    'pearson_p': stats['pearson_p'],
                    'spearman_rho': stats['spearman_rho'],
                    'spearman_p': stats['spearman_p'],
                    'r_squared': stats['r_squared'],
                    'variance_explained_pct': stats['r_squared'] * 100
                }

                # Add significance flags
                if stats['pearson_p'] < 0.001:
                    row['significance'] = '***'
                elif stats['pearson_p'] < 0.01:
                    row['significance'] = '**'
                elif stats['pearson_p'] < 0.05:
                    row['significance'] = '*'
                else:
                    row['significance'] = 'n.s.'

                rows.append(row)

    # Convert to DataFrame and save
    df = pd.DataFrame(rows)
    df = df.sort_values(['metric', 'dict_size'])
    df.to_csv(save_path, index=False, float_format='%.6f')

    print(f"💾 Saved correlation results (CSV) to: {save_path}")

    return df


def save_detailed_correlation_csv(all_results, layer):
    """Save detailed correlation data with individual author scores to CSV"""
    save_path = f'../data/diff_dict_size/correlation_detailed_all_metrics_layer_{layer}.csv'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for metric, results in all_results.items():
        for dict_size, data in sorted(results.items()):
            if data is not None:
                authors = data['authors']
                metric_scores = data['metric_scores']
                unlearn_difficulties = data['unlearning_difficulties']

                # Create a row for each author
                for author, metric_score, unlearn_diff in zip(authors, metric_scores, unlearn_difficulties):
                    row = {
                        'metric': metric,
                        'dict_size': dict_size,
                        'author': author,
                        'metric_score': metric_score,
                        'unlearning_difficulty': unlearn_diff
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(['metric', 'dict_size', 'author'])
    df.to_csv(save_path, index=False, float_format='%.6f')

    print(f"💾 Saved detailed data (CSV) to: {save_path}")

    return df


def save_summary_table_csv(all_results, layer):
    """Save the summary table in CSV format"""
    save_path = f'../data/diff_dict_size/correlation_detailed_all_metrics_layer_{layer}.csv'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    dict_sizes = [4096, 8192, 16384, 32768, 65536]
    metrics = list(all_results.keys())

    # Create rows for the summary table
    rows = []
    for metric in metrics:
        row = {'metric': metric}
        for dict_size in dict_sizes:
            if all_results[metric].get(dict_size) and all_results[metric][dict_size] is not None:
                r = all_results[metric][dict_size]['statistics']['pearson_r']
                p = all_results[metric][dict_size]['statistics']['pearson_p']

                # Significance marker
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = ""

                # Store r value and p value in separate columns
                row[f'{dict_size // 1000}k_r'] = r
                row[f'{dict_size // 1000}k_p'] = p
                row[f'{dict_size // 1000}k_sig'] = sig
            else:
                row[f'{dict_size // 1000}k_r'] = None
                row[f'{dict_size // 1000}k_p'] = None
                row[f'{dict_size // 1000}k_sig'] = 'N/A'

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, float_format='%.6f')

    print(f"💾 Saved summary table (CSV) to: {save_path}")

    return df


def plot_correlation_trends(all_results, save_path='../results/correlation_trends_all_metrics.png'):
    """
    Plot how correlation strength changes with dict_size for all metrics
    """

    metrics = list(all_results.keys())
    dict_sizes = [4096, 8192, 16384, 32768, 65536]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = {'jaccard': '#2E86AB', 'cosine': '#A23B72', 'l2': '#F18F01', 'l0': '#06A77D'}

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        results = all_results[metric]

        pearson_rs = []
        pearson_ps = []

        for dict_size in dict_sizes:
            if results.get(dict_size) and results[dict_size] is not None:
                pearson_rs.append(abs(results[dict_size]['statistics']['pearson_r']))
                pearson_ps.append(results[dict_size]['statistics']['pearson_p'])
            else:
                pearson_rs.append(np.nan)
                pearson_ps.append(np.nan)

        # Plot correlation strength
        ax.plot(dict_sizes, pearson_rs, 'o-', linewidth=2, markersize=8,
                color=colors.get(metric, 'gray'), label='|Pearson r|')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Moderate (0.5)')
        ax.axhline(y=0.7, color='darkred', linestyle='--', alpha=0.5, label='Strong (0.7)')

        # Mark significant points
        for i, (ds, r, p) in enumerate(zip(dict_sizes, pearson_rs, pearson_ps)):
            if p < 0.05:
                ax.scatter([ds], [r], s=200, facecolors='none', edgecolors='red', linewidths=3)

        ax.set_xlabel('Dictionary Size', fontsize=12)
        ax.set_ylabel('|Pearson r|', fontsize=12)
        ax.set_title(f'Metric: {metric.upper()}', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(dict_sizes)
        ax.set_xticklabels([f'{d // 1000}k' for d in dict_sizes])
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1])

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved trend plot to: {save_path}")

    #plt.show()


def plot_scatter_best_metric(all_results, save_path='../results/correlation_scatter_best.png'):
    """
    Plot scatter plots for the metric that shows the strongest correlation
    """

    # Find best metric (highest average |r| across dict_sizes)
    best_metric = None
    best_avg_r = 0

    for metric, results in all_results.items():
        avg_r = np.nanmean([abs(results[ds]['statistics']['pearson_r'])
                            for ds in results.keys() if results[ds] is not None])
        if avg_r > best_avg_r:
            best_avg_r = avg_r
            best_metric = metric

    print(f"\n🏆 Best metric: {best_metric} (avg |r| = {best_avg_r:.3f})")

    results = all_results[best_metric]
    dict_sizes = [k for k in sorted(results.keys()) if results[k] is not None]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, dict_size in enumerate(dict_sizes):
        if idx >= 6:
            break

        data = results[dict_size]
        ax = axes[idx]

        metric_scores = np.array(data['metric_scores'])
        unlearn = np.array(data['unlearning_difficulties'])
        authors = data['authors']
        stats_data = data['statistics']

        # Scatter plot
        ax.scatter(metric_scores, unlearn, alpha=0.6, s=100, color='#2E86AB')

        # Add regression line
        z = np.polyfit(metric_scores, unlearn, 1)
        p = np.poly1d(z)
        x_line = np.linspace(metric_scores.min(), metric_scores.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Annotate outliers (top 3 highest unlearning difficulty)
        top_indices = np.argsort(unlearn)[-3:]
        for i in top_indices:
            ax.annotate(authors[i], (metric_scores[i], unlearn[i]),
                        fontsize=8, alpha=0.7,
                        xytext=(5, 5), textcoords='offset points')

        # Labels and title
        ax.set_xlabel(f'{best_metric.upper()} Score', fontsize=11)
        ax.set_ylabel('Unlearning Difficulty', fontsize=11)

        sig = "***" if stats_data['pearson_p'] < 0.001 else "**" if stats_data['pearson_p'] < 0.01 else "*" if \
            stats_data['pearson_p'] < 0.05 else "n.s."
        title = f"Dict={dict_size // 1000}k: r={stats_data['pearson_r']:.2f}{sig}, R²={stats_data['r_squared']:.2f}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(dict_sizes), 6):
        axes[idx].axis('off')

    plt.suptitle(f'Best Metric: {best_metric.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved scatter plots to: {save_path}")

    #plt.show()


def print_summary_table(all_results):
    """Print a summary table of all metrics and dict_sizes"""

    print("\n" + "=" * 90)
    print("SUMMARY TABLE: Pearson r (p-value)")
    print("=" * 90)

    dict_sizes = [4096, 8192, 16384, 32768, 65536]
    metrics = list(all_results.keys())

    # Header
    header = f"{'Metric':<10}"
    for ds in dict_sizes:
        header += f"{ds // 1000:>8}k"
    print(header)
    print("-" * 90)

    # Rows
    for metric in metrics:
        row = f"{metric:<10}"
        for dict_size in dict_sizes:
            if all_results[metric].get(dict_size) and all_results[metric][dict_size] is not None:
                r = all_results[metric][dict_size]['statistics']['pearson_r']
                p = all_results[metric][dict_size]['statistics']['pearson_p']

                # Color code by significance
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = ""

                row += f"{r:+.2f}{sig:<3}"
            else:
                row += "     N/A"
        print(row)

    print("=" * 90)
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")


def plot_32k_l0_correlation(save_path='../results/l0_32k_correlation_highlight.png'):
    """Plot the significant correlation: 32k L0 vs Unlearning Difficulty"""

    # Load data
    superposition_data = load_superposition_data()
    unlearning_data = load_unlearning_results()

    # Extract 32k L0 data
    authors, l0_scores, unlearn_diff = extract_correlation_data(
        superposition_data, unlearning_data,
        dict_size=32768, layer_idx=11, threshold='0.01', metric='l0'
    )

    # Compute stats
    stats_result = compute_correlation_statistics(l0_scores, unlearn_diff)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot
    ax.scatter(l0_scores, unlearn_diff, alpha=0.7, s=150, color='#06A77D', edgecolors='black', linewidth=1)

    # Regression line
    z = np.polyfit(l0_scores, unlearn_diff, 1)
    p = np.poly1d(z)
    x_line = np.linspace(l0_scores.min(), l0_scores.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5, label='Linear fit')

    # Annotate ALL authors
    for i, author in enumerate(authors):
        ax.annotate(author, (l0_scores[i], unlearn_diff[i]),
                    fontsize=9, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')

    # Labels
    ax.set_xlabel('L0 Sparsity (# Active Features)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Unlearning Difficulty (Forget Loss Increase)', fontsize=14, fontweight='bold')
    ax.set_title(
        f'L0 Sparsity Predicts Unlearning Difficulty\n32k SAE, Layer 11: r={stats_result["pearson_r"]:.3f}, p={stats_result["pearson_p"]:.4f}*',
        fontsize=15, fontweight='bold')

    # Add stats box
    textstr = f'Pearson r = {stats_result["pearson_r"]:.3f}*\nP-value = {stats_result["pearson_p"]:.4f}\nR² = {stats_result["r_squared"]:.3f}\nN = {stats_result["n_samples"]}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved 32k L0 correlation plot to: {save_path}")
    plt.show()




def main():
    """Main analysis pipeline"""

    print("\n" + "=" * 70)
    print("PHASE 2: CORRELATION ANALYSIS - ALL METRICS")
    print("=" * 70)

    # Load data
    print("\n1) Loading data...")
    superposition_data = load_superposition_data()
    unlearning_data = load_unlearning_results()

    print(f"   ✓ Loaded superposition data for {len(superposition_data)} dict_sizes")
    print(f"   ✓ Loaded unlearning data for {len(unlearning_data)} authors")

    # Test all metrics
    metrics_to_test = ['jaccard', 'cosine', 'l2', 'l0']
    all_results = {}

    print("\n2) Computing correlations for all metrics...")
    for layer in range(12):
        for metric in metrics_to_test:
            print(f"\n{'#' * 70}")
            print(f"TESTING METRIC: {metric.upper()}")
            print(f"{'#' * 70}")

            results = analyze_all_dict_sizes(
                superposition_data,
                unlearning_data,
                layer_idx=layer,  # Layer 11
                threshold='0.01',
                metric=metric
            )

            all_results[metric] = results

        # Save results in multiple formats
        print("\n3) Saving results...")

        # JSON format (original)
        save_correlation_results(all_results, layer)

        # CSV formats (NEW)
        save_correlation_results_csv(all_results, layer)  # Summary statistics
        save_detailed_correlation_csv(all_results, layer)  # Individual author data
        save_summary_table_csv(all_results, layer)  # Summary table format

        # Print summary table
        print("\n4) Summary across all metrics...")
        print_summary_table(all_results)

        # Create visualizations
        print("\n5) Creating visualizations...")
        plot_correlation_trends(all_results)
        plot_scatter_best_metric(all_results)

        print("\n✅ Correlation analysis complete for all metrics!")


def create_author_ranking_table(save_path='../results/author_l0_ranking.csv'):
    """Create ranking of authors by L0 sparsity and difficulty"""

    superposition_data = load_superposition_data()
    unlearning_data = load_unlearning_results()

    authors, l0_scores, unlearn_diff = extract_correlation_data(
        superposition_data, unlearning_data,
        dict_size=32768, layer_idx=11, threshold='0.01', metric='l0'
    )

    # Create DataFrame
    df = pd.DataFrame({
        'author': authors,
        'l0_sparsity': l0_scores,
        'unlearning_difficulty': unlearn_diff
    })

    # Sort by L0 (high to low)
    df = df.sort_values('l0_sparsity', ascending=False)

    # Add rankings
    df['l0_rank'] = range(1, len(df) + 1)
    df['difficulty_rank'] = df['unlearning_difficulty'].rank().astype(int)

    # Categorize
    df['category'] = pd.cut(df['l0_sparsity'], bins=3, labels=['Low L0', 'Medium L0', 'High L0'])

    # Save
    df.to_csv(save_path, index=False, float_format='%.3f')
    print(f"\n📊 Saved author ranking to: {save_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("AUTHOR L0 RANKING (32k SAE, Layer 11)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("\n")

    # Print insights
    high_l0 = df.nlargest(5, 'l0_sparsity')
    low_l0 = df.nsmallest(5, 'l0_sparsity')

    print("📈 HIGH L0 (Distributed, Should be EASIER to unlearn):")
    print(high_l0[['author', 'l0_sparsity', 'unlearning_difficulty']].to_string(index=False))

    print("\n📉 LOW L0 (Concentrated, Should be HARDER to unlearn):")
    print(low_l0[['author', 'l0_sparsity', 'unlearning_difficulty']].to_string(index=False))

    return df


def analyze_author_specific_features(dict_size=32768, layer_idx=11, top_k=50):
    """
    For each author, identify:
    1. Author-specific features (high on author, low on retain)
    2. Shared features (high on both)

    Hypothesis: High-L0 authors have MORE author-specific features
    """

    superposition_data = load_superposition_data()

    sae_key = f'sae_{dict_size}'
    superposition_scores = superposition_data[sae_key]['superposition_scores']

    results = []

    for author in superposition_scores.keys():
        layer_key = str(layer_idx)

        # Get author activations and retain activations
        author_acts = np.array(superposition_scores[author][layer_key]['author_mean_activation'])
        retain_acts = np.array(superposition_scores[author][layer_key]['retain_mean_activation'])

        # Top-k author features
        top_author_features = np.argsort(author_acts)[-top_k:]

        # For each top author feature, check if it's shared with retain
        author_specific_count = 0
        shared_count = 0

        for feat_idx in top_author_features:
            author_val = author_acts[feat_idx]
            retain_val = retain_acts[feat_idx]

            # Define "author-specific" as: author >> retain (e.g., 3x higher)
            if author_val > 3 * retain_val:
                author_specific_count += 1
            # Define "shared" as: both high (retain > 0.5 * author)
            elif retain_val > 0.5 * author_val:
                shared_count += 1

        n_active = (author_acts > np.percentile(author_acts, 75)).sum()

        results.append({
            'author': author,
            'author_specific_features': author_specific_count,
            'shared_features': shared_count,
            'specificity_ratio_v1': author_specific_count / (author_specific_count + shared_count),  # Your current
            'specificity_ratio_v2': author_specific_count / n_active,  # Relative to L0
            'specificity_index': (author_specific_count - shared_count) / (author_specific_count + shared_count),  # Net specificity

        })

    df = pd.DataFrame(results)

    # Get L0 and difficulty for correlation
    authors, l0_scores, unlearn_diff = extract_correlation_data(
        load_superposition_data(), load_unlearning_results(),
        dict_size=dict_size, layer_idx=layer_idx, threshold='0.01', metric='l0'
    )

    l0_map = dict(zip(authors, l0_scores))
    diff_map = dict(zip(authors, unlearn_diff))

    df['l0_sparsity'] = df['author'].map(l0_map)
    df['unlearning_difficulty'] = df['author'].map(diff_map)

    # Correlate author-specific features with L0
    corr_l0 = stats.pearsonr(df['author_specific_features'], df['l0_sparsity'])
    print(f"\n📊 Correlation: Author-Specific Features vs L0: r={corr_l0[0]:.3f}, p={corr_l0[1]:.4f}")

    # Save
    df.to_csv('../results/author_feature_analysis.csv', index=False, float_format='%.3f')
    print(f"💾 Saved feature analysis to: ../results/author_feature_analysis.csv")

    return df





#def sae_guided_unlearning(author, dict_size=32768, layer_idx=11, ablation_strength=1.0):
    """
    SAE-guided unlearning: Ablate author-specific features

    Method:
    1. Identify top-k features for this author
    2. During inference, suppress these features
    3. Measure forget loss increase & retain preservation
    """

    # This requires integration with your unlearning pipeline
    # Pseudo-code:

    # 1. Load SAE for dict_size
    # sae = load_sae(dict_size, layer_idx)

    # 2. Get top-k author features
    # top_features = get_top_k_features(author, k=50)

    # 3. Create ablation hook
    # def ablation_hook(activations):
    #     sae_features = sae.encode(activations)
    #     sae_features[:, top_features] *= (1 - ablation_strength)  # Suppress
    #     return sae.decode(sae_features)

    # 4. Run evaluation with ablation
    # forget_loss = evaluate_forget_set(model, ablation_hook)
    # retain_loss = evaluate_retain_set(model, ablation_hook)

    # 5. Compare with gradient ascent baseline

    pass  # Implement this based on your unlearning code


def comprehensive_feature_analysis(author_acts, retain_acts, top_k=50):
    """
    Use multiple metrics to robustly identify author-specific features
    """

    results = {}

    # Method 1: Z-score (statistical)
    retain_std = np.std(retain_acts) + 1e-8
    z_scores = (author_acts - retain_acts) / retain_std
    author_specific_zscore = (z_scores > 2.0) & (author_acts > np.median(author_acts))
    results['method_1_count'] = author_specific_zscore.sum()

    # Method 2: Ratio (relative)
    ratio = author_acts / (retain_acts + 1e-8)
    top_author = author_acts > np.percentile(author_acts, 75)
    author_specific_ratio = top_author & (ratio > 10.0)
    results['method_2_count'] = author_specific_ratio.sum()

    # Method 3: Top-K overlap (structural)
    author_top_k = set(np.argsort(author_acts)[-top_k:])
    retain_top_k = set(np.argsort(retain_acts)[-top_k:])
    author_specific_topk = len(author_top_k - retain_top_k)
    shared_topk = len(author_top_k & retain_top_k)
    results['method_3_specific'] = author_specific_topk
    results['method_3_shared'] = shared_topk

    # Consensus: author-specific if 2+ methods agree
    author_specific_consensus = (
                                        author_specific_zscore.astype(int) +
                                        author_specific_ratio.astype(int)
                                ) >= 2

    results['consensus_count'] = author_specific_consensus.sum()
    results['consensus_indices'] = np.where(author_specific_consensus)[0]

    # Shared features: high on both (simple definition)
    shared = (author_acts > np.percentile(author_acts, 75)) & (retain_acts > np.percentile(retain_acts, 75))
    results['shared_count'] = shared.sum()

    # Specificity ratio
    total_high_features = (author_acts > np.percentile(author_acts, 75)).sum()
    results['specificity_ratio'] = results['consensus_count'] / total_high_features if total_high_features > 0 else 0

    return results

def analyze_author_specific_features_robust(layer_idx, dict_size=32768, top_k=50):
    """
    Robust analysis using multiple methods
    """

    superposition_data = load_superposition_data()

    sae_key = f'sae_{dict_size}'
    superposition_scores = superposition_data[sae_key]['superposition_scores']

    results = []

    for author in superposition_scores.keys():
        layer_key = str(layer_idx)

        # Get activations
        author_acts = np.array(superposition_scores[author][layer_key]['author_mean_activation'])
        retain_acts = np.array(superposition_scores[author][layer_key]['retain_mean_activation'])

        # Run comprehensive analysis
        feature_analysis = comprehensive_feature_analysis(author_acts, retain_acts, top_k)

        results.append({
            'author': author,
            'author_specific_consensus': feature_analysis['consensus_count'],
            'author_specific_zscore': feature_analysis['method_1_count'],
            'author_specific_ratio': feature_analysis['method_2_count'],
            'author_specific_topk': feature_analysis['method_3_specific'],
            'shared_features': feature_analysis['shared_count'],
            'specificity_ratio': feature_analysis['specificity_ratio'],
            'top_k_overlap': feature_analysis['method_3_shared']
        })

    df = pd.DataFrame(results)

    # Get L0 and difficulty
    authors, l0_scores, unlearn_diff = extract_correlation_data(
        load_superposition_data(), load_unlearning_results(),
        dict_size=dict_size, layer_idx=layer_idx, threshold='0.01', metric='l0'
    )

    l0_map = dict(zip(authors, l0_scores))
    diff_map = dict(zip(authors, unlearn_diff))

    df['l0_sparsity'] = df['author'].map(l0_map)
    df['unlearning_difficulty'] = df['author'].map(diff_map)

    # Correlate with L0 using CONSENSUS method
    corr_l0 = stats.pearsonr(df['author_specific_consensus'].dropna(),
                             df['l0_sparsity'].dropna())
    print(f"\n📊 Correlation: Author-Specific Features (Consensus) vs L0:")
    print(f"   r = {corr_l0[0]:.3f}, p = {corr_l0[1]:.4f}")

    # Also check specificity ratio
    corr_ratio = stats.pearsonr(df['specificity_ratio'].dropna(),
                                df['l0_sparsity'].dropna())
    print(f"\n📊 Correlation: Specificity Ratio vs L0:")
    print(f"   r = {corr_ratio[0]:.3f}, p = {corr_ratio[1]:.4f}")

    # Save
    df.to_csv(f'../results/author_feature_analysis_robust_layer_{layer_idx}.csv', index=False, float_format='%.3f')
    print(f"\n💾 Saved to: ../results/author_feature_analysis_robust.csv")
    print("\n\n")
    print(df)
    return df


if __name__ == "__main__":
    #main()

    # Call it
    #plot_32k_l0_correlation()

    # Call it
    # author_df = create_author_ranking_table()

    # Call it
    # feature_df = analyze_author_specific_features()
    for i in range(12):
        feature_df = analyze_author_specific_features_robust(i)





