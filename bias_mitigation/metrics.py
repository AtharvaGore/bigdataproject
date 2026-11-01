"""
Fairness and Performance Metrics on Apache Spark DataFrames.
Computes metrics in a single distributed aggregation query.
"""

from pyspark.sql import functions as F

def evaluate_fairness(df, protected_col="protected", label_col="label", prediction_col="prediction"):
    """
    Computes key fairness and classification metrics on a PySpark DataFrame:
    1. Statistical Parity Difference (SPD)
    2. Disparate Impact Ratio (DIR)
    3. Accuracy
    4. Group Acceptance Rates: P(Y_hat=1 | S=0) and P(Y_hat=1 | S=1)
    """
    # Distributed aggregation across all partitions
    agg_df = df.groupBy(protected_col).agg(
        F.count("*").alias("count"),
        F.sum(F.when(F.col(prediction_col) == 1.0, 1).otherwise(0)).alias("pred_favorable"),
        F.sum(F.when(F.col(prediction_col) == F.col(label_col), 1).otherwise(0)).alias("correct")
    )
    
    rows = agg_df.collect()
    
    # Map by protected group (0: Unprivileged, 1: Privileged)
    group_stats = {}
    total_samples = 0
    total_correct = 0
    
    for r in rows:
        g = int(r[protected_col])
        cnt = float(r["count"])
        fav = float(r["pred_favorable"])
        corr = float(r["correct"])
        
        group_stats[g] = {
            "count": cnt,
            "favorable": fav,
            "rate": fav / (cnt + 1e-12),
            "correct": corr
        }
        total_samples += cnt
        total_correct += corr
        
    p_unpriv = group_stats.get(0, {}).get("rate", 0.0)
    p_priv = group_stats.get(1, {}).get("rate", 0.0)
    
    # Statistical Parity Difference: P(Y_hat=1|S=0) - P(Y_hat=1|S=1)
    spd = p_unpriv - p_priv
    
    # Disparate Impact Ratio: P(Y_hat=1|S=0) / P(Y_hat=1|S=1)
    dir_ratio = p_unpriv / (p_priv + 1e-12)
    
    # Overall accuracy
    accuracy = total_correct / (total_samples + 1e-12)
    
    return {
        "accuracy": accuracy,
        "statistical_parity_diff": spd,
        "disparate_impact_ratio": dir_ratio,
        "unpriv_favorable_rate": p_unpriv,
        "priv_favorable_rate": p_priv,
        "total_count": int(total_samples)
    }

def print_metrics(metrics, method_name=""):
    """Pretty prints the metrics dictionary."""
    if method_name:
        print(f"\n--- {method_name} ---")
    print(f"  Accuracy:                   {metrics['accuracy'] * 100:.2f}%")
    print(f"  Disparate Impact Ratio:     {metrics['disparate_impact_ratio']:.4f}  (Ideal: >= 0.80, closer to 1.0)")
    print(f"  Statistical Parity Diff:    {metrics['statistical_parity_diff']:.4f}  (Ideal: 0.00)")
    print(f"  Unprivileged Favorable Rate: {metrics['unpriv_favorable_rate'] * 100:.2f}%")
    print(f"  Privileged Favorable Rate:   {metrics['priv_favorable_rate'] * 100:.2f}%")
