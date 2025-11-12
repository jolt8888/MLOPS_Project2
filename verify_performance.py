"""
Performance Verification Script
Compare training results between different environments (local, Docker, Colab)
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

def extract_metrics_from_checkpoint(checkpoint_path: str) -> Dict:
    """Extract metrics from checkpoint filename."""
    try:
        # Example: epoch=1-step=230-val_loss=0.3504.ckpt
        filename = Path(checkpoint_path).stem
        parts = filename.split('-')
        
        metrics = {}
        for part in parts:
            if '=' in part:
                key, value = part.split('=')
                try:
                    metrics[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    metrics[key] = value
        
        return metrics
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return {}

def list_checkpoints(checkpoint_dir: str = "checkpoints") -> List[Dict]:
    """List all checkpoints with their metrics."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    checkpoints = []
    for ckpt_file in checkpoint_path.glob("*.ckpt"):
        metrics = extract_metrics_from_checkpoint(str(ckpt_file))
        checkpoints.append({
            "filename": ckpt_file.name,
            "path": str(ckpt_file),
            "metrics": metrics
        })
    
    return sorted(checkpoints, key=lambda x: x['metrics'].get('val_loss', float('inf')))

def print_comparison_table(environments: Dict[str, Dict]):
    """Print a comparison table of results from different environments."""
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON ACROSS ENVIRONMENTS")
    print("="*80 + "\n")
    
    # Header
    print(f"{'Metric':<20} | {'Expected':<15} | {'Your Result':<15} | {'Difference':<15}")
    print("-" * 80)
    
    # Metrics to compare
    metrics = [
        ("Validation Loss", 0.350, "val_loss"),
        ("Accuracy", 0.846, "accuracy"),
        ("F1 Score", 0.893, "f1"),
    ]
    
    for metric_name, expected, key in metrics:
        your_result = environments.get('docker', {}).get(key, 'N/A')
        
        if isinstance(your_result, (int, float)):
            diff = your_result - expected
            diff_pct = (diff / expected) * 100
            diff_str = f"{diff:+.4f} ({diff_pct:+.1f}%)"
        else:
            diff_str = "N/A"
        
        print(f"{metric_name:<20} | {expected:<15.4f} | {str(your_result):<15} | {diff_str:<15}")
    
    print("-" * 80)
    print("\n")

def analyze_wandb_results(project_name: str = "glue-docker-training"):
    """Instructions for analyzing W&B results."""
    
    print("\n" + "="*80)
    print("W&B VERIFICATION CHECKLIST")
    print("="*80 + "\n")
    
    print("✓ Visit your W&B project: https://wandb.ai/")
    print(f"✓ Look for project: {project_name}")
    print(f"✓ Find run: best-hyperparams-docker\n")
    
    print("Check the following metrics:")
    print("  1. Final validation loss: Should be ~0.35-0.41")
    print("  2. Accuracy: Should be ~84-85%")
    print("  3. F1 Score: Should be ~0.89-0.90")
    print("  4. Training completed: All 3 epochs")
    print("  5. Loss curves: Smooth downward trend\n")
    
    print("Compare with local results:")
    print("  • Check if metrics are within ±2% of each other")
    print("  • Loss curves should follow similar patterns")
    print("  • Training time will differ (CPU vs GPU)")
    print("  • Exact numerical values may vary slightly due to:")
    print("    - Different hardware (CPU vs GPU precision)")
    print("    - Library version differences")
    print("    - Floating-point arithmetic variations\n")
    
    print("="*80 + "\n")

def main():
    """Main verification function."""
    
    print("\n" + "="*80)
    print("MLOPS PROJECT 2 - PERFORMANCE VERIFICATION")
    print("="*80 + "\n")
    
    # Check for checkpoints
    print("Analyzing checkpoints...\n")
    checkpoints = list_checkpoints()
    
    if not checkpoints:
        print("⚠ No checkpoints found. Have you run training yet?")
        print("  Run: .\\docker_run_wandb.ps1\n")
        return
    
    print(f"Found {len(checkpoints)} checkpoint(s):\n")
    
    # Display checkpoints
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"{i}. {ckpt['filename']}")
        for key, value in ckpt['metrics'].items():
            print(f"   {key}: {value}")
        print()
    
    # Best checkpoint analysis
    if checkpoints:
        best = checkpoints[0]
        print(f"Best checkpoint: {best['filename']}")
        
        # Extract metrics for comparison
        docker_results = best['metrics']
        
        # Expected results from previous training
        expected_results = {
            'val_loss': 0.3504,
            'accuracy': 0.846,
            'f1': 0.893
        }
        
        # Compare
        print("\nPerformance Comparison:")
        print("-" * 50)
        
        for metric, expected in expected_results.items():
            actual = docker_results.get(metric, 'N/A')
            if isinstance(actual, (int, float)):
                diff = actual - expected
                diff_pct = (diff / expected) * 100
                status = "✓" if abs(diff_pct) < 2 else "⚠"
                print(f"{status} {metric}: {actual:.4f} (expected: {expected:.4f}, diff: {diff:+.4f} / {diff_pct:+.1f}%)")
            else:
                print(f"⚠ {metric}: Not found in checkpoint name")
        
        print("-" * 50)
        
        # Verdict
        print("\nVerdict:")
        if all(metric in docker_results for metric in expected_results.keys()):
            diffs = [abs((docker_results[m] - expected_results[m]) / expected_results[m] * 100) 
                    for m in expected_results.keys()]
            if all(d < 2 for d in diffs):
                print("✓ ✓ ✓ EXCELLENT! Results are highly consistent with local training.")
                print("      Performance is within 2% tolerance.")
            elif all(d < 5 for d in diffs):
                print("✓ ✓  GOOD! Results are reasonably consistent.")
                print("     Small variations are normal due to hardware differences.")
            else:
                print("⚠  NOTICE: Results show some variation.")
                print("   This may be due to CPU vs GPU differences or library versions.")
        else:
            print("⚠  Could not verify all metrics from checkpoint filename.")
            print("   Check W&B dashboard for complete results.")
    
    # W&B instructions
    analyze_wandb_results()
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Check your W&B dashboard for detailed metrics and visualizations")
    print("2. Compare loss curves between local and Docker runs")
    print("3. Verify that hyperparameters match across environments")
    print("4. Document any performance differences in your report")
    print("5. Consider running on GitHub Codespaces or Docker Playground for cloud comparison")
    print("\n")

if __name__ == "__main__":
    main()
