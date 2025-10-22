#!/usr/bin/env python3
"""
Compare Baseline vs CAL Results

Reads experiment summaries and prints comparison table.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any

def load_experiment(exp_dir: Path) -> Dict[str, Any]:
    """Load experiment summary."""
    summary_path = exp_dir / "experiment_summary.json"
    
    if not summary_path.exists():
        return None
    
    with open(summary_path, 'r') as f:
        return json.load(f)

def print_comparison(baseline: Dict, cal: Dict):
    """Print formatted comparison table."""
    print("\n" + "="*80)
    print("BASELINE vs CAL-GRPO COMPARISON")
    print("="*80)
    print()
    
    # Training info
    print("Training Configuration:")
    print(f"  Model:        {baseline.get('model', 'N/A')}")
    print(f"  Dataset:      {baseline.get('dataset', 'N/A')}")
    print(f"  Train samples: {baseline.get('num_samples', 'N/A')}")
    print(f"  Batches:      {baseline.get('num_batches', 'N/A')}")
    print()
    
    # CAL config
    if cal and cal.get('use_cal'):
        print("CAL Configuration:")
        print(f"  CAL Model:    {cal.get('cal_model', 'N/A')}")
        print(f"  API Provider: {cal.get('api_provider', 'N/A')}")
        print(f"  Neg. Reward:  {cal.get('negative_reward', 'N/A')}")
        print()
    
    # Results comparison
    print("-"*80)
    print(f"{'Metric':<30} {'Baseline':<20} {'CAL':<20} {'Improvement':<10}")
    print("-"*80)
    
    # Accuracy
    baseline_acc = baseline.get('eval_accuracy', 0)
    cal_acc = cal.get('eval_accuracy', 0) if cal else 0
    improvement = cal_acc - baseline_acc if cal else 0
    improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
    
    print(f"{'Test Accuracy (%)':<30} {baseline_acc:<20.2f} {cal_acc:<20.2f} {improvement:+.2f} ({improvement_pct:+.1f}%)")
    
    # Num correct
    baseline_correct = baseline.get('eval_correct', 0)
    baseline_total = baseline.get('eval_total', 1)
    cal_correct = cal.get('eval_correct', 0) if cal else 0
    cal_total = cal.get('eval_total', 1) if cal else 1
    
    print(f"{'Correct / Total':<30} {baseline_correct}/{baseline_total:<14} {cal_correct}/{cal_total:<14}")
    
    print("-"*80)
    print()
    
    # Interpretation
    if cal:
        if improvement > 0:
            print(f"✅ CAL improves accuracy by {improvement:.2f} percentage points ({improvement_pct:+.1f}%)")
            print("   Your fine-grained credit assignment is working!")
        elif improvement < -1:
            print(f"⚠️  CAL decreased accuracy by {abs(improvement):.2f} percentage points")
            print("   Consider tuning: negative_reward, cal_model, or training samples")
        else:
            print("➖ CAL and baseline perform similarly")
            print("   Try: More training samples, different hyperparameters, or longer training")
    
    print("="*80)
    print()

def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs CAL results')
    parser.add_argument('--exp-dir', type=str, default='/tmp/cal_experiments',
                        help='Base experiments directory')
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    
    # Load experiments
    baseline = load_experiment(exp_dir / "baseline")
    cal = load_experiment(exp_dir / "cal")
    
    if not baseline:
        print(f"❌ No baseline results found at {exp_dir / 'baseline'}")
        print("   Run: python train_cal.py --num-samples 100")
        return
    
    if not cal:
        print(f"⚠️  No CAL results found at {exp_dir / 'cal'}")
        print("   Run: python train_cal.py --use-cal --num-samples 100")
        print()
        print("Showing baseline results only:")
        print()
    
    print_comparison(baseline, cal)
    
    # Detailed results
    print("\nDetailed Results:")
    print(f"  Baseline: {exp_dir / 'baseline'}")
    print(f"  CAL:      {exp_dir / 'cal'}")
    print()
    print("View training curves:")
    print(f"  tensorboard --logdir {exp_dir}/")
    print()

if __name__ == '__main__':
    main()

