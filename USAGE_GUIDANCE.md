# Usage Guidance & Project Reference
## Distributed Bias Mitigation on Apache Spark

---

### 1. Executive Summary & Problem Context
Traditional fairness toolkits (such as **AIF360** and **Fairlearn**) are designed for single-node machines using in-memory data structures (**Pandas**, **NumPy**, **Scikit-learn**). In enterprise production environments, data scales into millions of records (multi-gigabytes to terabytes). On a standard laptop, loading and processing these datasets causes **Out Of Memory (OOM) crashes**, system freezing, and thrashing.

This project delivers an end-to-end **Distributed Algorithmic Bias Mitigation Suite** built natively on **Apache Spark (PySpark & Spark MLlib)**. It implements the three canonical bias mitigation paradigms:
1. **Pre-processing**: **Distributed Reweighing** (Kamiran & Calders, 2012)
2. **In-processing**: **Distributed Fair Grid Search** (Cost-sensitive Reductions, Agarwal et al., 2018)
3. **Post-processing**: **Distributed Reject Option Classification (ROC)** (Kamiran, Karim, & Zhang, 2012)

---

### 2. Dataset Overview
Following the course specification, the project utilizes three datasets:

| Dataset | Sample Size | Target Label ($Y=1$) | Sensitive Attribute ($S$) | Purpose in Project |
| :--- | :--- | :--- | :--- | :--- |
| **UCI Adult Census** | 32,561 records | High Income (> $50K) | **Sex** (1: Male [Privileged], 0: Female [Unprivileged]) | **Correctness Verification** |
| **COMPAS (ProPublica)** | 5,278 records | Non-recidivism (No rearrest) | **Race** (1: Caucasian [Privileged], 0: African-American [Unprivileged]) | **Correctness Verification** |
| **Folktables (US ACS PUMS)** | Scaled from 50K to 1,000,000+ records | High Income (PINCP > $50K) | **Race / Sex** | **Scalability & Runtime Benchmarking** |

---

### 3. Understanding the Methods in Plain English

#### Method 1: Pre-processing (Distributed Reweighing)
* **When it runs**: **Before** model training (during the ETL / feature engineering stage).
* **The Intuition**: If a historical dataset is biased against women, female applicants who earned $> \$50\text{K}$ are rare. Reweighing assigns higher numerical sample weights to these underrepresented rows and lower weights to overrepresented rows so that the model pays equal attention to all demographic groups.
* **The Math**:
  $$W(s, y) = \frac{P(S=s) \times P(Y=y)}{P(S=s, Y=y)}$$
* **How Spark Scales It**: Spark computes the counts $N(s, y)$ in a single distributed `groupBy()` pass. The resulting scalar weights are applied via Spark SQL conditionals (zero network shuffle), and passed directly into `LogisticRegression(weightCol="reweight")`.

#### Method 2: In-processing (Distributed Fair Grid Search)
* **When it runs**: **During** model training (modifies the learning objective).
* **The Intuition**: Standard ML minimizes only classification error. Fair in-processing penalizes the model whenever it makes decisions with disparate impact. We test candidate penalty multipliers $\lambda \in [0.0, 1.0]$. As $\lambda$ increases, the cost of making false negatives on unprivileged applicants increases.
* **The Math**:
  $$\min_{\mathbf{w}} \mathcal{L}_{\text{classification}}(\mathbf{w}) + \lambda \cdot \text{DisparityPenalty}(\mathbf{w})$$
* **How Spark Scales It**: Runs parallel Spark MLlib distributed optimization iterations. Spark workers compute gradients across distributed partitions, while the driver selects the Pareto-optimal model with the best fairness-accuracy balance.

#### Method 3: Post-processing (Reject Option Classification - ROC)
* **When it runs**: **After** model training (black-box inference adjustment).
* **The Intuition**: A model outputs probability scores from $0.0$ to $1.0$. Around the decision boundary ($0.50 \pm \theta$), the model is highly uncertain (borderline cases). In this uncertainty zone, we give unprivileged applicants the benefit of the doubt (round up to $1.0$) and round borderline privileged applicants down to $0.0$.
* **The Math**:
  $$\hat{Y}_{\text{mitigated}} = \begin{cases} 
  1 & \text{if } S=0 \text{ and } P(Y=1|X) \in [0.5 - \theta, 0.5) \\
  0 & \text{if } S=1 \text{ and } P(Y=1|X) \in [0.5, 0.5 + \theta] \\
  \hat{Y}_{\text{original}} & \text{otherwise}
  \end{cases}$$
* **How Spark Scales It**: The threshold $\theta$ is tuned on a validation split, and transformation executes via a single Spark SQL `F.when().otherwise()` expression with zero retraining and instant execution over millions of rows.

---

### 4. Metrics Reference Guide

1. **Disparate Impact Ratio (DIR)**:
   $$\text{DIR} = \frac{P(\hat{Y}=1 \mid S=\text{unprivileged})}{P(\hat{Y}=1 \mid S=\text{privileged})}$$
   * **Rule of Thumb**: Legal standard (**EEOC 80% Rule**) requires $\text{DIR} \ge 0.80$. Ideal is $1.00$. Below $0.80$ is legally biased.
2. **Statistical Parity Difference (SPD)**:
   $$\text{SPD} = P(\hat{Y}=1 \mid S=\text{unprivileged}) - P(\hat{Y}=1 \mid S=\text{privileged})$$
   * **Rule of Thumb**: Measures acceptance rate gap. Ideal is $0.00$. Negative values mean the unprivileged group is accepted at lower rates.
3. **Accuracy**:
   Percentage of overall correct predictions on the held-out test split.

---

### 5. Step-by-Step Execution Guide

Every script can be executed directly inside the project virtual environment:

```bash
# Activate environment (or use ./venv/bin/python)
cd /home/atharva/programs/bigdata/proj

# 1. Download datasets (Adult, COMPAS, Folktables ACS PUMS)
./venv/bin/python data/download_datasets.py

# 2. Run Correctness Benchmarks (UCI Adult & COMPAS)
./venv/bin/python run_correctness.py

# 3. Run Scalability Benchmarks (Folktables at 50K, 100K, 250K, 500K, 1M rows)
./venv/bin/python run_scalability.py

# 4. Generate Visualization Charts
./venv/bin/python plot_results.py
```

All generated plots and JSON reports are saved in `output/`:
- `output/scalability_curves.png`: Runtime vs. $N$ and Throughput vs. $N$ charts.
- `output/fairness_accuracy_tradeoff.png`: Before/after fairness comparison charts.
- `output/correctness_results.json`: Raw numerical metrics for Adult & COMPAS.
- `output/scalability_results.json`: Raw scalability timings across sizes.

---

### 6. Real-World Decision Matrix

When should an enterprise use each method?

| Method | Computational Cost | Retraining Required? | Model Access Required? | Recommended When |
| :--- | :--- | :--- | :--- | :--- |
| **Pre-processing (Reweighing)** | **Very Low** ($O(N)$ single pass) | Yes | Yes (needs `weightCol`) | Data pipeline / ETL stages where raw data can be modified and downstream models accept sample weights. |
| **In-processing (Fair Grid)** | **Moderate** ($K \times \text{train time}$) | Yes | Yes (internal loss) | Maximum fairness-accuracy optimization is required and training resources are available. |
| **Post-processing (ROC)** | **Near Zero** (instant SQL transform) | **No** | **No** (Black-box compatible) | Working with proprietary, legacy, or third-party black-box models where retraining is impossible or cost-prohibitive. |
