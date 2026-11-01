# Distributed Algorithmic Bias Mitigation on Apache Spark

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5%20%2B-orange)](https://spark.apache.org/)

> **Case Study**: Traditional machine learning fairness toolkits (such as IBM's **AIF360** and Microsoft's **Fairlearn**) run on a single laptop using in-memory Python structures (`pandas`, `numpy`, `scikit-learn`). When enterprise datasets exceed available RAM, these toolkits crash with **Out-Of-Memory (OOM)** errors.
>
> This repository implements all three canonical bias mitigation paradigms natively on **Apache Spark (PySpark & Spark MLlib)**, evaluating correctness, runtime, throughput, and scalability up to **1,000,000+ records**.

---

## 1. Architecture Diagram

```mermaid
flowchart TD
    subgraph Data Layer
        A["Input Data Sources<br/>(Folktables, COMPAS, UCI Adult)"] --> B["PySpark Distributed DataFrame"]
        B --> F["Feature Assembler & One-Hot Encoder"]
    end

    subgraph Bias Mitigation Paradigms on Spark
        direction TB
        
        subgraph Pre-processing
            C1["Distributed Reweighing<br/>W(s,y) = P(s)*P(y) / P(s,y)"]
            C1 --> C2["Weighted Spark DataFrame"]
            C2 --> C3["Standard Spark MLlib Classifier<br/>LogisticRegression(weightCol='reweight')"]
        end

        subgraph In-processing
            D1["Fair Grid Search<br/>Loss + lambda * DisparityPenalty"]
            D1 --> D2["Iterative Dual Optimization<br/>over lambda in [0.0, 1.0]"]
            D2 --> D3["Pareto-Optimal Fair Model"]
        end

        subgraph Post-processing
            E1["Unmitigated Base Model<br/>Predicted Probabilities P(Y=1|X)"]
            E1 --> E2["Reject Option Classification (ROC)<br/>Uncertainty Band [0.5 - theta, 0.5 + theta]"]
            E2 --> E3["SQL-level Flip Logic via Spark F.when()"]
        end
    end

    subgraph Evaluation & Outputs
        G["Distributed Fairness Evaluator<br/>(Accuracy, Disparate Impact, Parity Difference)"]
        H["Scalability Curves & Trade-Off Plots<br/>output/scalability_curves.png"]
        I["Interactive Results Notebook<br/>results_notebook.ipynb"]
    end

    F --> C1
    F --> D1
    F --> E1
    C3 --> G
    D3 --> G
    E3 --> G
    G --> H
    G --> I
```

---

## 2. Repository Structure

```
.
├── bias_mitigation/            # Core Spark fairness implementations
│   ├── spark_utils.py          # SparkSession builder with Java 21 support
│   ├── pipeline_utils.py       # Spark ML VectorAssembler & OneHotEncoder pipeline
│   ├── metrics.py              # Distributed Disparate Impact & Parity Difference evaluator
│   ├── preprocessing.py        # Method 1: Distributed Reweighing (Kamiran & Calders, 2012)
│   ├── inprocessing.py         # Method 2: Fair Grid Search (Agarwal et al., 2018)
│   └── postprocessing.py       # Method 3: Reject Option Classification (Kamiran et al., 2012)
├── data/
│   └── download_datasets.py    # Auto-downloads Adult, COMPAS, and Folktables (ACS PUMS)
├── output/                     # Benchmark results and publication-grade plots
│   ├── correctness_results.json
│   ├── scalability_results.json
│   ├── scalability_curves.png
│   └── fairness_accuracy_tradeoff.png
├── results_notebook.ipynb      # Interactive Jupyter results notebook
├── run_correctness.py          # Runs correctness evaluation on UCI Adult & COMPAS
├── run_scalability.py          # Runs scalability evaluation on Folktables up to 1,000,000 rows
├── plot_results.py             # Generates visualization charts
├── run_pipeline.sh             # One-click reproducible execution script
├── Dockerfile                  # Containerized environment for full reproducibility
├── docker-compose.yml          # Docker Compose configuration
├── USAGE_GUIDANCE.md           # Detailed user guide & plain-English concept walkthrough
├── requirements.txt            # Locked Python package dependencies
├── .gitignore                  # Prevents committing large data archives to GitHub
└── README.md
```

---

## 3. Quick Setup & Reproducible Pipeline

The pipeline downloads datasets on demand, so large data files do not need to be committed to Git. Reproduce a run by recording the source versions, environment, Spark settings, and generated outputs.

### Option A: Local Execution (Recommended)

#### Prerequisites:
* **Python**: 3.10, 3.11, or 3.12
* **Java**: OpenJDK 17 or 21 (required by PySpark 3.5+)

#### Run the Entire Pipeline in 1 Command:
```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Run full pipeline (downloads datasets, runs benchmarks, generates plots)
./run_pipeline.sh
```

> **Tip for a fast demo**: Run `./run_pipeline.sh --quick` to test scales up to 100,000 rows in under 2 minutes.

---

### Option B: Docker Compose Execution

If you prefer complete container isolation without configuring Java or Python locally:

```bash
docker compose up --build
```
This automatically builds the container, installs dependencies, downloads datasets, and writes all charts and results directly to the mounted `output/` folder.

---

## 4. Key Benchmark Results

### A. Correctness Verification (UCI Adult & COMPAS)

Evaluated on held-out test splits comparing the Unmitigated Baseline against the 3 bias mitigation methods:

#### 1. UCI Adult Census Income (32,561 rows — Sensitive: Gender)
| Method | Accuracy | Disparate Impact Ratio | Statistical Parity Diff | Evaluation |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Unmitigated)** | 84.54% | **0.3203** | -0.1670 | Severe historical gender disparity |
| **Pre-processing (Reweighing)** | 84.09% | **0.6090** | -0.0787 | Disparity cut by > 50% |
| **In-processing (Fair Grid Search)** | 82.69% | **0.9090** | **-0.0122** | DIR above 0.80; SPD is close to zero |
| **Post-processing (ROC)** | 83.06% | **0.7479** | -0.0382 | Fairness improved; assess the accuracy trade-off |

#### 2. COMPAS Recidivism (5,278 rows — Sensitive: Race)
| Method | Accuracy | Disparate Impact Ratio | Statistical Parity Diff | Evaluation |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Unmitigated)** | 67.06% | **0.6446** | -0.2682 | Unfair baseline favoring privileged group |
| **Pre-processing (Reweighing)** | 66.80% | **0.6506** | -0.2637 | Slight improvement |
| **In-processing (Fair Grid Search)** | 62.63% | **0.8168** | -0.1725 | DIR above 0.80 screening threshold |
| **Post-processing (ROC)** | 66.41% | **0.9503** | **-0.0317** | Lower absolute SPD; evaluate task-specific trade-offs |

---

### B. Scalability Benchmarks (Folktables US ACS PUMS up to 1,000,000 rows)

Runtime and throughput scaling evaluated on distributed Spark vs. single-node laptop execution:

| Dataset Size ($N$) | Spark Reweighing Runtime (Throughput) | Spark In-Processing Runtime (Throughput) | Spark ROC Runtime (Throughput) | Single-Node Laptop (Pandas / Sklearn) |
| :---: | :---: | :---: | :---: | :---: |
| **50,000 rows** | 12.286s (4,070 rows/s) | 31.606s (1,582 rows/s) | 8.058s (6,205 rows/s) | Measured: 0.634s (7.75 MB traced peak) |
| **100,000 rows** | 6.715s (14,892 rows/s) | 24.937s (4,010 rows/s) | 6.377s (15,682 rows/s) | Measured: 3.192s (13.92 MB traced peak) |
| **250,000 rows** | 6.805s (36,736 rows/s) | 24.957s (10,017 rows/s) | 7.102s (35,203 rows/s) | Measured: 5.587s (34.82 MB traced peak) |
| **500,000 rows** | 7.611s (65,697 rows/s) | 27.587s (18,125 rows/s) | 7.796s (64,139 rows/s) | Not run; OOM status and 400 MB are estimates |
| **1,000,000 rows** | 8.382s (119,300 rows/s) | 30.683s (32,591 rows/s) | 9.028s (110,766 rows/s) | Not run; OOM status and 800 MB are estimates |


**Benchmark notes:** Runtime values above are copied from `output/scalability_results.json`. The laptop baseline runs only through 250,000 rows; larger laptop OOM entries are estimates because the script skips those runs. The scalability script samples the source for smaller sizes and repeats rows for sizes above the available source. Repeated records stress the pipeline but do not represent independent population growth. See [RESULTS_CORRECTIONS.md](RESULTS_CORRECTIONS.md) for details.

**Data-use note:** UCI Adult, COMPAS, and Folktables ACS are public person-level records. Confirm they satisfy the course's restriction on real personal data; if not, replace them with synthetic data and rerun the experiments. Public availability alone does not make person-level records synthetic or anonymous. See [DATA_CARD.md](DATA_CARD.md).

---

## 5. Visualizations

### Scalability Curves (Runtime & Throughput vs. Data Size $N$)

The checked-in scalability and fairness-accuracy figures were regenerated from the committed JSON with `python plot_results.py`. Rerun that command after updating the result files.

![Scalability Curves](output/scalability_curves.png)

### Fairness vs. Accuracy Trade-Off
![Fairness vs Accuracy](output/fairness_accuracy_tradeoff.png)

---

## 6. Interactive Results Notebook

To inspect and interact with the results, open the Jupyter Notebook:
```bash
./venv/bin/jupyter notebook results_notebook.ipynb
```
The notebook contains:
* Interactive tables of all experiment outcomes.
* Embedded high-resolution plots.
* Step-by-step mathematical explanations and code walkthroughs.

---

## 7. Production Usage Guidance

When should an enterprise choose each method?

| Method | Overhead | Retraining Needed? | Model Invasiveness | Best Used When |
| :--- | :--- | :---: | :--- | :--- |
| **Pre-processing (Reweighing)** | **Minimal** ($O(N)$ single pass) | Yes | None (any weighted model) | You control the ETL pipeline and want instant data debiasing before model fitting. |
| **In-processing (Fair Grid)** | **Moderate** ($K \times \text{train time}$) | Yes | High (modifies loss objective) | You need mathematical guarantees and the highest fairness-accuracy Pareto frontier. |
| **Post-processing (ROC)** | **Near Zero** (instant SQL transform) | **No** | None (black-box model) | Working with frozen, third-party, or legacy black-box models where retraining is not permitted. |

For complete documentation and theoretical derivations, see [USAGE_GUIDANCE.md](USAGE_GUIDANCE.md).
