"""
Plotting Script for Distributed Bias Mitigation
Generates two high-quality figures:
1. output/scalability_curves.png (Runtime & Throughput vs Data Size N)
2. output/fairness_accuracy_tradeoff.png (Fairness vs Accuracy on Adult & COMPAS)
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_scalability(out_dir):
    json_path = os.path.join(out_dir, "scalability_results.json")
    if not os.path.exists(json_path):
        print(f"Scalability results not found at {json_path}")
        return
        
    with open(json_path) as f:
        data = json.load(f)
        
    scales = sorted([int(k) for k in data.keys()])
    scale_labels = [f"{s//1000}K" if s < 1000000 else f"{s//1000000}M" for s in scales]
    
    t_rw = [data[str(s)]["spark_preprocessing_reweighing"]["runtime_sec"] for s in scales]
    t_inproc = [data[str(s)]["spark_inprocessing_fair_grid"]["runtime_sec"] for s in scales]
    t_roc = [data[str(s)]["spark_postprocessing_roc"]["runtime_sec"] for s in scales]
    
    tp_rw = [data[str(s)]["spark_preprocessing_reweighing"]["throughput_rows_per_sec"] for s in scales]
    tp_inproc = [data[str(s)]["spark_inprocessing_fair_grid"]["throughput_rows_per_sec"] for s in scales]
    tp_roc = [data[str(s)]["spark_postprocessing_roc"]["throughput_rows_per_sec"] for s in scales]
    
    # Single-node runtimes
    sn_times = []
    for s in scales:
        single_node = data[str(s)]["single_node_laptop"]
        # Plot only completed measurements; estimated OOM entries are not observations.
        val = single_node["runtime_sec"] if single_node.get("status") == "SUCCESS" else None
        sn_times.append(val)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # ----------------- SUBPLOT 1: RUNTIME (LOG-LOG) -----------------
    ax1.plot(scales, t_rw, marker="o", linewidth=2.5, color="#1f77b4", label="Spark Pre-processing (Reweighing)")
    ax1.plot(scales, t_inproc, marker="s", linewidth=2.5, color="#ff7f0e", label="Spark In-processing (Fair Grid)")
    ax1.plot(scales, t_roc, marker="^", linewidth=2.5, color="#2ca02c", label="Spark Post-processing (ROC)")
    
    # Single node valid points
    valid_sn_x = [s for s, t in zip(scales, sn_times) if t is not None]
    valid_sn_y = [t for t in sn_times if t is not None]
    if valid_sn_x:
        ax1.plot(valid_sn_x, valid_sn_y, marker="x", linewidth=2.2, linestyle="--", color="#d62728", label="Single-Node Laptop (Pandas/Sklearn)")
        
    ax1.text(0.98, 0.04, "Laptop baseline not measured above 250K", transform=ax1.transAxes,
             ha="right", va="bottom", color="#555555", fontsize=9)
    
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Dataset Size (Number of Rows N)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Execution Time (Seconds, Log Scale)", fontsize=12, fontweight="bold")
    ax1.set_title("Runtime Scalability on Spark vs. Single-Node Laptop", fontsize=13, fontweight="bold")
    ax1.set_xticks(scales)
    ax1.set_xticklabels(scale_labels)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # ----------------- SUBPLOT 2: THROUGHPUT (ROWS / SEC) -----------------
    ax2.plot(scales, tp_rw, marker="o", linewidth=2.5, color="#1f77b4", label="Pre-processing (Reweighing)")
    ax2.plot(scales, tp_inproc, marker="s", linewidth=2.5, color="#ff7f0e", label="In-processing (Fair Grid)")
    ax2.plot(scales, tp_roc, marker="^", linewidth=2.5, color="#2ca02c", label="Post-processing (ROC)")
    
    ax2.set_xscale("log")
    ax2.set_xlabel("Dataset Size (Number of Rows N)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Throughput (Rows Processed / Second)", fontsize=12, fontweight="bold")
    ax2.set_title("Distributed Spark Throughput Scaling", fontsize=13, fontweight="bold")
    ax2.set_xticks(scales)
    ax2.set_xticklabels(scale_labels)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plot_file = os.path.join(out_dir, "scalability_curves.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved scalability curves to {plot_file}")

def plot_fairness_accuracy(out_dir):
    json_path = os.path.join(out_dir, "correctness_results.json")
    if not os.path.exists(json_path):
        print(f"Correctness results not found at {json_path}")
        return
        
    with open(json_path) as f:
        data = json.load(f)
        
    datasets = ["UCI_Adult", "COMPAS"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    methods = ["Baseline", "Pre-processing (Reweighing)", "In-processing (Fair Grid Search)", "Post-processing (ROC)"]
    labels = ["Baseline", "Pre-processing\n(Reweighing)", "In-processing\n(Fair Grid)", "Post-processing\n(ROC)"]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for ax, ds in zip(axes, datasets):
        ds_data = data[ds]
        accuracies = [ds_data[m]["accuracy"] * 100 for m in methods]
        disparate_impacts = [ds_data[m]["disparate_impact_ratio"] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        # Dual axis: left for Accuracy, right for Disparate Impact Ratio
        ax2 = ax.twinx()
        
        rects1 = ax.bar(x - width/2, accuracies, width, label="Accuracy (%)", color="#4a90e2", alpha=0.85)
        rects2 = ax2.bar(x + width/2, disparate_impacts, width, label="Disparate Impact Ratio", color="#50e3c2", alpha=0.85)
        
        # 80% legal rule threshold line for Disparate Impact
        ax2.axhline(0.80, color="#d62728", linestyle="--", linewidth=1.8, label="80% Fairness Rule (DIR >= 0.8)")
        ax2.axhline(1.00, color="#2ca02c", linestyle=":", linewidth=1.5, label="Perfect Parity (DIR = 1.0)")
        
        ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold", color="#1c4e80")
        ax2.set_ylabel("Disparate Impact Ratio (DIR)", fontsize=11, fontweight="bold", color="#008080")
        ax.set_ylim(50, 100)
        ax2.set_ylim(0, 1.25)
        
        ds_title = "UCI Adult Census Income" if ds == "UCI_Adult" else "COMPAS Recidivism"
        ax.set_title(f"Fairness vs. Accuracy: {ds_title}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        
        # Annotate values on top of bars
        for r in rects1:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for r in rects2:
            h = r.get_height()
            ax2.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                         
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left", fontsize=9, frameon=True)
        ax.grid(False)
        ax2.grid(False)
        
    plt.tight_layout()
    plot_file = os.path.join(out_dir, "fairness_accuracy_tradeoff.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved fairness vs accuracy plot to {plot_file}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "output")
    plot_scalability(out_dir)
    plot_fairness_accuracy(out_dir)
