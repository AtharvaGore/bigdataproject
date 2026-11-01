# Three minute demo recording script

Record the actual application and terminal output. Do not present estimated laptop OOM entries as observed crashes. If the dataset policy has not been cleared, record the demo using a fully synthetic dataset instead of the person-level datasets currently configured.

## 0:00–0:25 — Problem and architecture

**Show:** repository README architecture diagram.

**Say:** “This project applies preprocessing, in-processing, and post-processing fairness mitigation with Spark DataFrames. Spark distributes data and model work across partitions. We compare predictive performance and group fairness as the dataset size increases.”

## 0:25–0:55 — Setup and reproducibility

**Show:** Docker Compose and pipeline instructions, then the output folder.

**Say:** “The pipeline downloads the configured datasets, runs correctness and scalability experiments, and writes JSON results and plots. These outputs are checked in so the reported numbers can be inspected.”

## 0:55–1:35 — Three mitigation methods

**Show:** `bias_mitigation/preprocessing.py`, `inprocessing.py`, and `postprocessing.py` in sequence.

**Say:** “Reweighing calculates group-and-label weights before model fitting. The in-processing method trains candidate models and selects using validation fairness and accuracy. Reject-option post-processing changes borderline predictions using the protected group. Each method has trade-offs; none guarantees fairness in every context.”

## 1:35–2:20 — Evidence

**Show:** `output/correctness_results.json`, then the results notebook and plots.

**Say:** “The correctness results report accuracy, statistical parity difference, and disparate impact on held-out splits. The scalability output reports runtimes across five row counts. Runtime values vary between runs. The repository's README table must match the raw JSON before we present the figures.”

## 2:20–2:45 — Scaling limits

**Show:** `output/scalability_results.json` entries for 250K, 500K, and 1M.

**Say:** “The checked-in JSON measures the laptop baseline through 250,000 rows. The larger laptop entries are estimated OOM labels because the script skips those runs. Large Spark inputs are generated partly by repeating source rows, so these figures are runtime stress tests, not evidence of independent population-scale fairness.”

## 2:45–3:00 — Responsible use and close

**Show:** `DATA_CARD.md`.

**Say:** “The datasets contain person-level public records. We are confirming that their use satisfies the course's no-real-personal-data rule. A dataset card records the source, transformations, sensitive attributes, and limitations. Fairness metrics are diagnostic measures, not proof of legal compliance.”

