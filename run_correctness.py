"""
Correctness Evaluation Script
Evaluates Baseline, Pre-processing (Reweighing), In-processing (Fair Grid Search),
and Post-processing (Reject Option Classification) on UCI Adult and COMPAS datasets.
"""

import os
import json
from pyspark.ml.classification import LogisticRegression

from bias_mitigation.spark_utils import get_spark_session
from bias_mitigation.metrics import evaluate_fairness, print_metrics
from bias_mitigation.preprocessing import DistributedReweighing
from bias_mitigation.inprocessing import DistributedFairInProcessing
from bias_mitigation.postprocessing import DistributedRejectOptionClassifier
from bias_mitigation.pipeline_utils import build_feature_pipeline

def evaluate_dataset(spark, dataset_name, csv_path, feature_cols, cat_cols):
    print(f"\n{'='*70}")
    print(f" EVALUATING DATASET: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Load data into Spark DataFrame
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    total_count = df.count()
    print(f"Loaded {total_count} rows from {csv_path}")
    
    # Build & apply feature pipeline
    pipe = build_feature_pipeline(df, feature_cols, cat_cols)
    transformed_df = pipe.transform(df).select("features", "label", "protected")
    
    # Train / Val / Test split (70% train, 15% val, 15% test)
    train_df, val_df, test_df = transformed_df.randomSplit([0.70, 0.15, 0.15], seed=42)
    train_df = train_df.cache()
    val_df = val_df.cache()
    test_df = test_df.cache()
    
    results = {}
    
    # -------------------------------------------------------------------------
    # 1. BASELINE (UNMITIGATED)
    # -------------------------------------------------------------------------
    print("\n[1/4] Training Baseline (Unmitigated Logistic Regression)...")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50, regParam=0.01)
    base_model = lr.fit(train_df)
    base_test_preds = base_model.transform(test_df)
    base_metrics = evaluate_fairness(base_test_preds, "protected", "label", "prediction")
    print_metrics(base_metrics, "Baseline (Unmitigated)")
    results["Baseline"] = base_metrics
    
    # -------------------------------------------------------------------------
    # 2. PRE-PROCESSING (REWEIGHING)
    # -------------------------------------------------------------------------
    print("\n[2/4] Training Pre-processing (Distributed Reweighing)...")
    reweighing = DistributedReweighing(protected_col="protected", label_col="label")
    rw_model, _ = reweighing.fit_model(train_df, features_col="features")
    rw_test_preds = rw_model.transform(test_df)
    rw_metrics = evaluate_fairness(rw_test_preds, "protected", "label", "prediction")
    print_metrics(rw_metrics, "Pre-processing (Reweighing)")
    results["Pre-processing (Reweighing)"] = rw_metrics
    
    # -------------------------------------------------------------------------
    # 3. IN-PROCESSING (FAIR GRID SEARCH)
    # -------------------------------------------------------------------------
    print("\n[3/4] Training In-processing (Fair Grid Search)...")
    inproc = DistributedFairInProcessing(protected_col="protected", label_col="label")
    inproc_model, best_lam, _ = inproc.fit(train_df, val_df, features_col="features")
    print(f"  Optimal fairness penalty lambda selected: {best_lam}")
    inproc_test_preds = inproc_model.transform(test_df)
    inproc_metrics = evaluate_fairness(inproc_test_preds, "protected", "label", "prediction")
    print_metrics(inproc_metrics, "In-processing (Fair Grid Search)")
    results["In-processing (Fair Grid Search)"] = inproc_metrics
    
    # -------------------------------------------------------------------------
    # 4. POST-PROCESSING (REJECT OPTION CLASSIFICATION)
    # -------------------------------------------------------------------------
    print("\n[4/4] Applying Post-processing (Reject Option Classification)...")
    base_val_preds = base_model.transform(val_df)
    roc = DistributedRejectOptionClassifier(protected_col="protected", label_col="label")
    best_theta, _ = roc.fit(base_val_preds, prob_col="probability", pred_col="prediction")
    print(f"  Optimal uncertainty band margin theta selected: {best_theta}")
    roc_test_preds = roc.transform(base_test_preds, prob_col="probability", pred_col="prediction", out_col="mitigated_pred")
    roc_metrics = evaluate_fairness(roc_test_preds, "protected", "label", "mitigated_pred")
    print_metrics(roc_metrics, "Post-processing (ROC)")
    results["Post-processing (ROC)"] = roc_metrics
    
    # Summary Table
    print("\n" + "-"*75)
    print(f"{'Method':<32} | {'Accuracy':<10} | {'Disparate Impact':<18} | {'Parity Diff':<12}")
    print("-"*75)
    for m_name, m in results.items():
        print(f"{m_name:<32} | {m['accuracy']*100:6.2f}%    | {m['disparate_impact_ratio']:8.4f}           | {m['statistical_parity_diff']:8.4f}")
    print("-"*75)
    
    return results

def main():
    spark = get_spark_session("CorrectnessBenchmarks")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    all_results = {}
    
    # 1. UCI Adult
    adult_csv = os.path.join(base_dir, "data", "adult.csv")
    adult_features = [
        "age", "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "capital_gain", "capital_loss", "hours_per_week"
    ]
    adult_cats = ["workclass", "education", "marital_status", "occupation", "relationship", "race"]
    all_results["UCI_Adult"] = evaluate_dataset(spark, "UCI Adult Census", adult_csv, adult_features, adult_cats)
    
    # 2. COMPAS
    compas_csv = os.path.join(base_dir, "data", "compas.csv")
    compas_features = ["age", "priors_count", "decile_score", "c_charge_degree", "sex"]
    compas_cats = ["c_charge_degree", "sex"]
    all_results["COMPAS"] = evaluate_dataset(spark, "COMPAS Recidivism", compas_csv, compas_features, compas_cats)
    
    # Save results to JSON
    out_json = os.path.join(out_dir, "correctness_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll correctness benchmark results saved to {out_json}")
    
    spark.stop()

if __name__ == "__main__":
    main()
