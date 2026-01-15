"""
Comparison Harness: Single Model vs Dual Model Weekend Normalization

Runs both approaches and generates comparative analysis.
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def run_approach(script_name, approach_name):
    """Run an approach and capture results."""
    print(f"\n{'='*80}")
    print(f"Running: {approach_name}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(
        ['python', script_name],
        capture_output=True,
        text=True,
        cwd='.'
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def load_results():
    """Load results from both approaches."""
    single_path = Path('reports/results/anomaly_summary.json')
    dual_path = Path('reports/results/anomaly_summary_dual.json')
    
    single = json.load(open(single_path)) if single_path.exists() else None
    dual = json.load(open(dual_path)) if dual_path.exists() else None
    
    return single, dual

def compare_and_visualize(single, dual):
    """Generate comparison visualizations and report."""
    if not single or not dual:
        print("⚠️ Missing results files!")
        return
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Comparison table
    comparison = pd.DataFrame({
        'Metric': [
            'Total Anomalies',
            'Weekday Anomalies',
            'Weekend Anomalies',
            'Weekday Anomaly Rate (%)',
            'Weekend Anomaly Rate (%)',
            'Weekday Threshold',
            'Weekend Threshold',
            'Weekend FP Reduction'
        ],
        'Single Model (Dual Thresholds)': [
            single['total_anomalies'],
            single['weekday_anomalies'],
            single['weekend_anomalies'],
            f"{100*single['weekday_anomalies']/single['weekday_samples']:.1f}",
            f"{100*single['weekend_anomalies']/single['weekend_samples']:.1f}",
            f"{single['weekday_threshold']:.4f}",
            f"{single['weekend_threshold']:.4f}",
            single.get('false_positive_reduction', 'N/A')
        ],
        'Dual Model (Separate AEs)': [
            dual['total_anomalies'],
            dual['weekday_anomalies'],
            dual['weekend_anomalies'],
            f"{100*dual['weekday_anomalies']/dual['weekday_samples']:.1f}",
            f"{100*dual['weekend_anomalies']/dual['weekend_samples']:.1f}",
            f"{dual['weekday_threshold']:.4f}",
            f"{dual['weekend_threshold']:.4f}",
            'N/A'
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Save comparison table
    out_dir = Path('reports/results')
    comparison.to_csv(out_dir / 'weekend_normalization_comparison.csv', index=False)
    print(f"\n✅ Comparison saved to {out_dir / 'weekend_normalization_comparison.csv'}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Anomaly counts
    categories = ['Weekday', 'Weekend']
    single_counts = [single['weekday_anomalies'], single['weekend_anomalies']]
    dual_counts = [dual['weekday_anomalies'], dual['weekend_anomalies']]
    
    x = range(len(categories))
    width = 0.35
    
    axes[0].bar([i - width/2 for i in x], single_counts, width, label='Single Model', alpha=0.8, color='steelblue')
    axes[0].bar([i + width/2 for i in x], dual_counts, width, label='Dual Model', alpha=0.8, color='coral')
    axes[0].set_ylabel('Anomaly Count')
    axes[0].set_title('Anomaly Counts by Approach')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Anomaly rates
    single_rates = [
        100*single['weekday_anomalies']/single['weekday_samples'],
        100*single['weekend_anomalies']/single['weekend_samples']
    ]
    dual_rates = [
        100*dual['weekday_anomalies']/dual['weekday_samples'],
        100*dual['weekend_anomalies']/dual['weekend_samples']
    ]
    
    axes[1].bar([i - width/2 for i in x], single_rates, width, label='Single Model', alpha=0.8, color='steelblue')
    axes[1].bar([i + width/2 for i in x], dual_rates, width, label='Dual Model', alpha=0.8, color='coral')
    axes[1].set_ylabel('Anomaly Rate (%)')
    axes[1].set_title('Anomaly Detection Rates by Approach')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].axhline(5, color='red', linestyle='--', alpha=0.5, label='Target 5%')
    
    plt.tight_layout()
    plt.savefig('reports/figures/modeling/weekend_normalization_comparison.png', dpi=150)
    print(f"✅ Visualization saved to reports/figures/modeling/weekend_normalization_comparison.png")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    single_weekend_rate = 100*single['weekend_anomalies']/single['weekend_samples']
    dual_weekend_rate = 100*dual['weekend_anomalies']/dual['weekend_samples']
    
    if abs(single_weekend_rate - dual_weekend_rate) < 0.5:
        print("📊 RESULT: Both approaches perform similarly")
        print("💡 RECOMMENDATION: Use SINGLE MODEL (simpler, less storage, faster deployment)")
        recommendation = "single_model"
    elif single_weekend_rate < dual_weekend_rate:
        print("📊 RESULT: Single model has lower weekend false positive rate")
        print("💡 RECOMMENDATION: Use SINGLE MODEL")
        recommendation = "single_model"
    else:
        print("📊 RESULT: Dual model has lower weekend false positive rate")
        print("💡 RECOMMENDATION: Use DUAL MODEL")
        recommendation = "dual_model"
    
    # Save recommendation
    rec_data = {
        'recommendation': recommendation,
        'reasoning': f"Single weekend rate: {single_weekend_rate:.1f}%, Dual weekend rate: {dual_weekend_rate:.1f}%",
        'comparison': comparison.to_dict('records')
    }
    
    with open(out_dir / 'weekend_recommendation.json', 'w') as f:
        json.dump(rec_data, f, indent=2)

def main():
    print("Weekend Normalization Comparison Experiment")
    print("="*80)
    
    # Run single model approach
    success1 = run_approach(
        'src/analysis/modeling/08_autoencoder.py',
        'Single Model with Dual Thresholds'
    )
    
    # Run dual model approach
    success2 = run_approach(
        'src/analysis/modeling/08_autoencoder_dual.py',
        'Dual Model (Separate Autoencoders)'
    )
    
    if success1 and success2:
        # Load and compare results
        single, dual = load_results()
        compare_and_visualize(single, dual)
    else:
        print("\n⚠️ One or both approaches failed. Check error messages above.")

if __name__ == "__main__":
    main()
