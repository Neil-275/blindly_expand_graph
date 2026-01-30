"""
Utility functions for subgraph coverage experiments
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import torch

def calculate_query_coverage(query, sampling_method, method_name):
        """Calculate coverage metrics for a single query"""
    # subgraph = None
    # try:
        if method_name == 'bfs':
            topk_nodes, _, subgraph = sampling_method.sampleSubgraphBFS(query)
        else:
            topk_nodes, _, subgraph = sampling_method.sampleSubgraph(query)
        # Handle answers
        answers = set(query.get('answers_id', []))
        answer = query['answers_id'][0]
        if len(answers) == 0:
            # If no answers, coverage is undefined, set to 0
            coverage = 0.0
            hit = 0
        else:
            topk_node_set = set(topk_nodes.tolist()) if topk_nodes is not None else set()
            # print(topk_node_set)
            intersection = answers & topk_node_set
            coverage = len(intersection) / len(answers)
            hit = 1 if len(intersection) > 0 else 0
        # print("type subgraph: " + str(type(subgraph)))
        # print("type(topk_nodes): " + str(type(topk_nodes)))
        subgraph_size = len(torch.concat([subgraph[:,0], subgraph[:,2]]).unique()) if subgraph is not None else 0
        
        return {
            'coverage': coverage,
            'hit': hit,
            'subgraph_size': subgraph_size,
            'num_answers': len(answers)
        }
        
    # except Exception as e:
        # # Return default values if there's an error
        # # print(subgraph)
        # print(f"Error in calculate_query_coverage: {str(e)}")
        
        # return {
        #     'coverage': 0.0,
        #     'hit': 0,
        #     'subgraph_size': 0,
        #     'num_answers': 0
        # }

def test_method_on_queries(queries, sampling_method, method_name):
    """Test a method on a set of queries"""
    
    if len(queries) == 0:
        return {
            'mean_coverage': 0.0,
            'hit_rate': 0.0,
            'avg_subgraph_size': 0.0,
            'num_queries': 0
        }
    
    total_coverage = 0 
    total_hits = 0
    subgraph_sizes = []
    successful_queries = 0
    skipped_queries = []
    for i, query in enumerate(queries):
        # try:
            metrics = calculate_query_coverage(query, sampling_method, method_name)
            subgraph_size = metrics['subgraph_size']
            if subgraph_size == 0:
                skipped_queries.append(i)
                print(f"\n     ⚠️ Skipping query {i+1} due to empty subgraph.")
                continue
            total_coverage += metrics['coverage']
            total_hits += metrics['hit']
            subgraph_sizes.append(subgraph_size)
            successful_queries += 1
            
        # except Exception as e:
        #     print(f"\n     ⚠️ Error processing query {i+1}: {str(e)}. Skipping...")
        #     skipped_queries.append(i)
        #     continue
    
    if successful_queries == 0:
        return {
            'mean_coverage': 0.0,
            'hit_rate': 0.0,
            'avg_subgraph_size': 0.0,
            'num_queries': 0
        }
    
    return {
        'mean_coverage': total_coverage / successful_queries,
        'hit_rate': total_hits / successful_queries,
        'avg_subgraph_size': np.mean(subgraph_sizes) if subgraph_sizes else 0.0,
        'num_queries': successful_queries,
        'skipped_queries': skipped_queries
    }



def create_results_table(results_df, hyperparameter='cands_lim'):
    """Create a formatted results table"""
    
    # Determine which metric column to use
    metric_col = 'mean_coverage' if 'mean_coverage' in results_df.columns else 'mean_precision'
    
    # Set up the table title based on hyperparameter
    if hyperparameter == 'cands_lim':
        title = "CANDIDATES LIMIT PERFORMANCE COMPARISON"
        param_name = "Candidates Limit"
    elif hyperparameter == 'fact_ratio':
        title = "FACT RATIO PERFORMANCE COMPARISON"
        param_name = "Fact Ratio"
    else:
        title = f"{hyperparameter.upper()} PERFORMANCE COMPARISON"
        param_name = hyperparameter
    
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)
    
    # Check if hyperparameter column exists
    if hyperparameter not in results_df.columns:
        print(f"⚠️ Warning: '{hyperparameter}' column not found in results.")
        print(f"Available columns: {list(results_df.columns)}")
        return None
    
    try:
        # Format the table
        table = results_df.pivot_table(
            index='query_type', 
            columns=hyperparameter, 
            values=[metric_col, 'hit_rate', 'avg_subgraph_size'],
            aggfunc='first'
        )
        
        print(f"\n🎯 Mean Coverage/Precision:")
        print(table[metric_col].round(4).to_string())
        
        print(f"\n🎯 Hit Rate:")
        print(table['hit_rate'].round(4).to_string())
        
        print(f"\n📏 Average Subgraph Size:")
        print(table['avg_subgraph_size'].round(1).to_string())
        
        return table
        
    except Exception as e:
        print(f"⚠️ Error creating pivot table: {str(e)}")
        print("\n📊 Raw Results Summary:")
        print(results_df.round(4).to_string(index=False))
        return results_df


def plot_fact_ratio_cands_lim_analysis(results_df, save_path=None):
    """
    Create comprehensive plots analyzing the effect of fact_ratio and cands_lim 
    on coverage and hit rate metrics.
    
    Args:
        results_df: DataFrame with columns ['fact_ratio', 'cands_lim', 'query_type', 
                   'mean_coverage', 'hit_rate', 'avg_subgraph_size']
        save_path: Optional path to save the plots
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Subgraph Retrieval Analysis: Impact of Fact Ratio and Candidates Limit', 
                 fontsize=16, fontweight='bold')
    
    # Color palette for query types
    query_colors = {'1_hop': '#FF6B6B', '2_hop': '#4ECDC4', '3_hop': '#45B7D1'}
    
    # 1. Coverage vs Candidates Limit (by fact_ratio)
    ax1 = axes[0, 0]
    for fact_ratio in sorted(results_df['fact_ratio'].unique()):
        subset = results_df[results_df['fact_ratio'] == fact_ratio]
        for query_type in sorted(subset['query_type'].unique()):
            data = subset[subset['query_type'] == query_type]
            ax1.plot(data['cands_lim'], data['mean_coverage'], 
                    marker='o', linewidth=2, markersize=6,
                    color=query_colors[query_type], 
                    alpha=0.7 if fact_ratio != 0.6 else 1.0,
                    linestyle='-' if fact_ratio == 0.4 else '--' if fact_ratio == 0.6 else ':',
                    label=f'{query_type} (ratio={fact_ratio})' if fact_ratio == 0.6 else "")
    
    ax1.set_xlabel('Candidates Limit', fontweight='bold')
    ax1.set_ylabel('Mean Coverage', fontweight='bold')
    ax1.set_title('Coverage vs Candidates Limit\n(Different line styles = different fact ratios)', 
                  fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Hit Rate vs Candidates Limit (by fact_ratio)
    ax2 = axes[0, 1]
    for fact_ratio in sorted(results_df['fact_ratio'].unique()):
        subset = results_df[results_df['fact_ratio'] == fact_ratio]
        for query_type in sorted(subset['query_type'].unique()):
            data = subset[subset['query_type'] == query_type]
            ax2.plot(data['cands_lim'], data['hit_rate'], 
                    marker='s', linewidth=2, markersize=6,
                    color=query_colors[query_type],
                    alpha=0.7 if fact_ratio != 0.6 else 1.0,
                    linestyle='-' if fact_ratio == 0.4 else '--' if fact_ratio == 0.6 else ':')
    
    ax2.set_xlabel('Candidates Limit', fontweight='bold')
    ax2.set_ylabel('Hit Rate', fontweight='bold')
    ax2.set_title('Hit Rate vs Candidates Limit\n(Different line styles = different fact ratios)', 
                  fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Coverage vs Fact Ratio (by cands_lim)
    ax3 = axes[0, 2]
    for cands_lim in sorted(results_df['cands_lim'].unique()):
        subset = results_df[results_df['cands_lim'] == cands_lim]
        for query_type in sorted(subset['query_type'].unique()):
            data = subset[subset['query_type'] == query_type]
            ax3.plot(data['fact_ratio'], data['mean_coverage'], 
                    marker='o', linewidth=2, markersize=6,
                    color=query_colors[query_type],
                    alpha=0.7 if cands_lim != 512 else 1.0,
                    linestyle='-' if cands_lim == 256 else '--' if cands_lim == 512 else ':',
                    label=f'{query_type} (lim={cands_lim})' if cands_lim == 512 else "")
    
    ax3.set_xlabel('Fact Ratio', fontweight='bold')
    ax3.set_ylabel('Mean Coverage', fontweight='bold')
    ax3.set_title('Coverage vs Fact Ratio\n(Different line styles = different cands limits)', 
                  fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Heatmap: Coverage by Fact Ratio and Candidates Limit (averaged across query types)
    ax4 = axes[1, 0]
    pivot_coverage = results_df.groupby(['fact_ratio', 'cands_lim'])['mean_coverage'].mean().unstack()
    sns.heatmap(pivot_coverage, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Coverage'})
    ax4.set_title('Average Coverage Heatmap\n(Averaged across query types)', fontweight='bold')
    ax4.set_xlabel('Candidates Limit', fontweight='bold')
    ax4.set_ylabel('Fact Ratio', fontweight='bold')
    
    # 5. Heatmap: Hit Rate by Fact Ratio and Candidates Limit (averaged across query types)
    ax5 = axes[1, 1]
    pivot_hit = results_df.groupby(['fact_ratio', 'cands_lim'])['hit_rate'].mean().unstack()
    sns.heatmap(pivot_hit, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax5, cbar_kws={'label': 'Hit Rate'})
    ax5.set_title('Average Hit Rate Heatmap\n(Averaged across query types)', fontweight='bold')
    ax5.set_xlabel('Candidates Limit', fontweight='bold')
    ax5.set_ylabel('Fact Ratio', fontweight='bold')
    
    # 6. 3D-like plot showing relationship between all three variables
    ax6 = axes[1, 2]
    for query_type in sorted(results_df['query_type'].unique()):
        subset = results_df[results_df['query_type'] == query_type]
        scatter = ax6.scatter(subset['fact_ratio'], subset['cands_lim'], 
                             c=subset['mean_coverage'], s=subset['hit_rate']*200,
                             alpha=0.7, cmap='viridis', 
                             label=query_type, edgecolors='black', linewidth=0.5)
    
    ax6.set_xlabel('Fact Ratio', fontweight='bold')
    ax6.set_ylabel('Candidates Limit', fontweight='bold')
    ax6.set_title('Coverage & Hit Rate Combined\n(Color=Coverage, Size=Hit Rate)', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add colorbar for the last plot
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Mean Coverage', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()


def plot_query_type_comparison(results_df, metric='mean_coverage', save_path=None):
    """
    Create detailed comparison plots for different query types.
    
    Args:
        results_df: DataFrame with experimental results
        metric: Metric to plot ('mean_coverage', 'hit_rate', 'avg_subgraph_size')
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{metric.replace("_", " ").title()} Comparison Across Query Types', 
                 fontsize=14, fontweight='bold')
    
    query_types = sorted(results_df['query_type'].unique())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, query_type in enumerate(query_types):
        subset = results_df[results_df['query_type'] == query_type]
        
        # Create a pivot table for this query type
        pivot = subset.pivot(index='fact_ratio', columns='cands_lim', values=metric)
        
        # Plot as line plot
        ax = axes[i]
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker='o', linewidth=2, 
                   markersize=6, label=f'cands_lim={col}')
        
        ax.set_title(f'{query_type}', fontweight='bold')
        ax.set_xlabel('Fact Ratio', fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()


def plot_performance_trends(results_df, save_path=None):
    """
    Create plots showing performance trends and trade-offs.
    
    Args:
        results_df: DataFrame with experimental results
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Trends and Trade-offs Analysis', fontsize=16, fontweight='bold')
    
    # 1. Coverage vs Hit Rate scatter
    ax1 = axes[0, 0]
    for query_type in sorted(results_df['query_type'].unique()):
        subset = results_df[results_df['query_type'] == query_type]
        ax1.scatter(subset['mean_coverage'], subset['hit_rate'], 
                   alpha=0.7, s=60, label=query_type)
    
    ax1.set_xlabel('Mean Coverage', fontweight='bold')
    ax1.set_ylabel('Hit Rate', fontweight='bold')
    ax1.set_title('Coverage vs Hit Rate Trade-off', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Efficiency plot: Coverage per subgraph node
    ax2 = axes[0, 1]
    results_df['efficiency'] = results_df['mean_coverage'] / (results_df['avg_subgraph_size'] / 1000)
    for query_type in sorted(results_df['query_type'].unique()):
        subset = results_df[results_df['query_type'] == query_type]
        ax2.plot(subset['cands_lim'], subset['efficiency'], 
                marker='o', linewidth=2, label=query_type)
    
    ax2.set_xlabel('Candidates Limit', fontweight='bold')
    ax2.set_ylabel('Coverage per 1K Nodes', fontweight='bold')
    ax2.set_title('Efficiency: Coverage per Subgraph Size', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot of coverage distribution
    ax3 = axes[1, 0]
    coverage_data = []
    labels = []
    for cands_lim in sorted(results_df['cands_lim'].unique()):
        subset = results_df[results_df['cands_lim'] == cands_lim]
        coverage_data.append(subset['mean_coverage'])
        labels.append(f'{cands_lim}')
    
    ax3.boxplot(coverage_data, labels=labels)
    ax3.set_xlabel('Candidates Limit', fontweight='bold')
    ax3.set_ylabel('Mean Coverage Distribution', fontweight='bold')
    ax3.set_title('Coverage Distribution by Candidates Limit', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Fact ratio impact analysis
    ax4 = axes[1, 1]
    fact_ratio_impact = results_df.groupby(['fact_ratio', 'query_type']).agg({
        'mean_coverage': 'mean',
        'hit_rate': 'mean'
    }).reset_index()
    
    for query_type in sorted(fact_ratio_impact['query_type'].unique()):
        subset = fact_ratio_impact[fact_ratio_impact['query_type'] == query_type]
        ax4.plot(subset['fact_ratio'], subset['mean_coverage'], 
                marker='o', linewidth=2, label=f'{query_type} Coverage')
        ax4.plot(subset['fact_ratio'], subset['hit_rate'], 
                marker='s', linewidth=2, linestyle='--', label=f'{query_type} Hit Rate')
    
    ax4.set_xlabel('Fact Ratio', fontweight='bold')
    ax4.set_ylabel('Performance Metric', fontweight='bold')
    ax4.set_title('Fact Ratio Impact on Performance', fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()


def generate_analysis_report(results_df, save_path=None):
    """
    Generate a comprehensive analysis report with statistics and insights.
    
    Args:
        results_df: DataFrame with experimental results
        save_path: Optional path to save the report
    """
    report = []
    report.append("="*80)
    report.append("🔍 COMPREHENSIVE SUBGRAPH RETRIEVAL ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # Basic statistics
    report.append("📊 DATASET SUMMARY:")
    report.append(f"   • Total experiments: {len(results_df)}")
    report.append(f"   • Fact ratio range: {results_df['fact_ratio'].min():.1f} - {results_df['fact_ratio'].max():.1f}")
    report.append(f"   • Candidates limit range: {results_df['cands_lim'].min()} - {results_df['cands_lim'].max()}")
    report.append(f"   • Query types: {', '.join(sorted(results_df['query_type'].unique()))}")
    report.append("")
    
    # Performance analysis
    report.append("🎯 PERFORMANCE ANALYSIS:")
    best_coverage = results_df.loc[results_df['mean_coverage'].idxmax()]
    best_hit_rate = results_df.loc[results_df['hit_rate'].idxmax()]
    
    report.append(f"   • Best Coverage: {best_coverage['mean_coverage']:.3f}")
    report.append(f"     └─ Configuration: fact_ratio={best_coverage['fact_ratio']}, cands_lim={best_coverage['cands_lim']}, query_type={best_coverage['query_type']}")
    report.append(f"   • Best Hit Rate: {best_hit_rate['hit_rate']:.3f}")
    report.append(f"     └─ Configuration: fact_ratio={best_hit_rate['fact_ratio']}, cands_lim={best_hit_rate['cands_lim']}, query_type={best_hit_rate['query_type']}")
    report.append("")
    
    # Trend analysis
    report.append("📈 TREND ANALYSIS:")
    
    # Candidates limit effect
    cands_effect = results_df.groupby('cands_lim').agg({
        'mean_coverage': 'mean',
        'hit_rate': 'mean'
    })
    report.append(f"   • Candidates Limit Effect:")
    for cands_lim in sorted(results_df['cands_lim'].unique()):
        cov = cands_effect.loc[cands_lim, 'mean_coverage']
        hit = cands_effect.loc[cands_lim, 'hit_rate']
        report.append(f"     └─ {cands_lim}: Coverage={cov:.3f}, Hit Rate={hit:.3f}")
    
    # Fact ratio effect
    fact_effect = results_df.groupby('fact_ratio').agg({
        'mean_coverage': 'mean',
        'hit_rate': 'mean'
    })
    report.append(f"   • Fact Ratio Effect:")
    for fact_ratio in sorted(results_df['fact_ratio'].unique()):
        cov = fact_effect.loc[fact_ratio, 'mean_coverage']
        hit = fact_effect.loc[fact_ratio, 'hit_rate']
        report.append(f"     └─ {fact_ratio}: Coverage={cov:.3f}, Hit Rate={hit:.3f}")
    report.append("")
    
    # Query complexity analysis
    report.append("🔗 QUERY COMPLEXITY ANALYSIS:")
    query_stats = results_df.groupby('query_type').agg({
        'mean_coverage': ['mean', 'std'],
        'hit_rate': ['mean', 'std'],
        'avg_subgraph_size': 'mean'
    }).round(3)
    
    for query_type in sorted(results_df['query_type'].unique()):
        stats = query_stats.loc[query_type]
        report.append(f"   • {query_type}:")
        report.append(f"     └─ Coverage: {stats[('mean_coverage', 'mean')]:.3f} ± {stats[('mean_coverage', 'std')]:.3f}")
        report.append(f"     └─ Hit Rate: {stats[('hit_rate', 'mean')]:.3f} ± {stats[('hit_rate', 'std')]:.3f}")
        report.append(f"     └─ Avg Subgraph Size: {stats[('avg_subgraph_size', 'mean')]:.1f}")
    report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS:")
    
    # Find optimal configurations
    avg_metrics = results_df.groupby(['fact_ratio', 'cands_lim']).agg({
        'mean_coverage': 'mean',
        'hit_rate': 'mean'
    })
    
    # Combined score (equally weighted)
    combined_score = (avg_metrics['mean_coverage'] + avg_metrics['hit_rate']) / 2
    best_config = combined_score.idxmax()
    
    report.append(f"   • Optimal Configuration (balanced): fact_ratio={best_config[0]}, cands_lim={best_config[1]}")
    report.append(f"   • For maximum coverage: fact_ratio={best_coverage['fact_ratio']}, cands_lim={best_coverage['cands_lim']}")
    report.append(f"   • For maximum hit rate: fact_ratio={best_hit_rate['fact_ratio']}, cands_lim={best_hit_rate['cands_lim']}")
    
    # Efficiency recommendations
    results_df['efficiency'] = results_df['mean_coverage'] / (results_df['avg_subgraph_size'] / 1000)
    most_efficient = results_df.loc[results_df['efficiency'].idxmax()]
    report.append(f"   • Most efficient (coverage/size): fact_ratio={most_efficient['fact_ratio']}, cands_lim={most_efficient['cands_lim']}")
    
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"📄 Report saved to: {save_path}")
    
    return report_text


def analyze_results_from_csv(csv_path, output_dir='results/expand_subgraph'):
    """
    Convenience function to load CSV data and run all analyses
    
    Args:
        csv_path: Path to the CSV file with results
        output_dir: Directory to save plots and reports
    
    Returns:
        DataFrame with the loaded results
    """
    import os
    
    # Load data
    results_df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(results_df)} experimental results from {csv_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    plot_fact_ratio_cands_lim_analysis(
        results_df, 
        save_path=f"{output_dir}/comprehensive_analysis.png"
    )
    
    plot_query_type_comparison(
        results_df, 
        metric='mean_coverage',
        save_path=f"{output_dir}/coverage_by_query_type.png"
    )
    
    plot_query_type_comparison(
        results_df, 
        metric='hit_rate',
        save_path=f"{output_dir}/hit_rate_by_query_type.png"
    )
    
    plot_performance_trends(
        results_df,
        save_path=f"{output_dir}/performance_trends.png"
    )
    
    # Generate report
    generate_analysis_report(
        results_df,
        save_path=f"{output_dir}/analysis_report.txt"
    )
    
    print(f"✅ All analyses completed! Results saved to {output_dir}/")
    
    return results_df

# ++++++++++++++++
# Usage functions
# ++++++++++++++++
def run_complete_analysis(csv_path='results/expand_subgraph/cands_lim_results.csv'):
    """
    Run complete analysis of subgraph retrieval experiments
    
    Args:
        csv_path: Path to the CSV file containing experimental results
    """
    print("🔍 Loading experimental results...")
    
    # Load the data
    try:
        results_df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(results_df)} experimental results")
        print(f"   Columns: {list(results_df.columns)}")
        print(f"   Fact ratios tested: {sorted(results_df['fact_ratio'].unique())}")
        print(f"   Candidates limits tested: {sorted(results_df['cands_lim'].unique())}")
        print(f"   Query types: {sorted(results_df['query_type'].unique())}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {csv_path}")
        return
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    print("\n" + "="*60)
    print("🎨 GENERATING VISUALIZATION PLOTS...")
    print("="*60)
    
    # 1. Comprehensive analysis plot
    print("\n📊 Creating comprehensive fact_ratio & cands_lim analysis...")
    try:
        plot_fact_ratio_cands_lim_analysis(
            results_df, 
            save_path='results/expand_subgraph/comprehensive_analysis.png'
        )
        print("✅ Comprehensive analysis plot created successfully")
    except Exception as e:
        print(f"❌ Error creating comprehensive plot: {str(e)}")
    
    # 2. Query type comparison for coverage
    print("\n📈 Creating query type comparison for coverage...")
    try:
        plot_query_type_comparison(
            results_df, 
            metric='mean_coverage',
            save_path='results/expand_subgraph/query_type_coverage_comparison.png'
        )
        print("✅ Query type coverage comparison created successfully")
    except Exception as e:
        print(f"❌ Error creating query type coverage plot: {str(e)}")
    
    # 3. Query type comparison for hit rate
    print("\n🎯 Creating query type comparison for hit rate...")
    try:
        plot_query_type_comparison(
            results_df, 
            metric='hit_rate',
            save_path='results/expand_subgraph/query_type_hitrate_comparison.png'
        )
        print("✅ Query type hit rate comparison created successfully")
    except Exception as e:
        print(f"❌ Error creating query type hit rate plot: {str(e)}")
    
    # 4. Performance trends and trade-offs
    print("\n⚖️ Creating performance trends analysis...")
    try:
        plot_performance_trends(
            results_df,
            save_path='results/expand_subgraph/performance_trends.png'
        )
        print("✅ Performance trends analysis created successfully")
    except Exception as e:
        print(f"❌ Error creating performance trends plot: {str(e)}")
    
    print("\n" + "="*60)
    print("📝 GENERATING ANALYSIS REPORT...")
    print("="*60)
    
    # 5. Generate comprehensive report
    try:
        generate_analysis_report(
            results_df,
            save_path='results/expand_subgraph/analysis_report.txt'
        )
        print("✅ Analysis report generated successfully")
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
    
    print("\n🎉 Analysis complete! Check the results/expand_subgraph/ folder for all generated files.")
    
    return results_df

def quick_analysis_example():
    """
    Quick example showing how to create individual plots
    """
    # Load data
    results_df = pd.read_csv('results/expand_subgraph/cands_lim_results.csv')
    
    # Create just one type of plot
    print("Creating a quick comprehensive analysis plot...")
    plot_fact_ratio_cands_lim_analysis(results_df)
    
    # Show some basic statistics
    print("\n📊 Quick Statistics:")
    print(f"Mean coverage across all experiments: {results_df['mean_coverage'].mean():.3f}")
    print(f"Mean hit rate across all experiments: {results_df['hit_rate'].mean():.3f}")
    
    # Best performing configurations
    best_coverage_config = results_df.loc[results_df['mean_coverage'].idxmax()]
    print(f"\n🏆 Best coverage configuration:")
    print(f"   Fact ratio: {best_coverage_config['fact_ratio']}")
    print(f"   Candidates limit: {best_coverage_config['cands_lim']}")
    print(f"   Query type: {best_coverage_config['query_type']}")
    print(f"   Coverage: {best_coverage_config['mean_coverage']:.3f}")
    
    return results_df