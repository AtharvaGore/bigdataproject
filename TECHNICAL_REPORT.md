# Distributed Bias Mitigation on Apache Spark

## Technical report

### Executive summary

This project implements three fairness mitigation stages with PySpark: reweighing before training, a validation-selected fair model during training, and reject-option prediction adjustment after training. It evaluates accuracy and group outcome rates on UCI Adult and COMPAS, then benchmarks Spark processing at five input sizes using Folktables ACS data. The repository includes code, raw result JSON, plots, a notebook, container setup, and usage guidance. The committed Spark results show multi-second processing. The laptop comparison measures only through 250,000 rows; the OOM labels at 500,000 and 1,000,000 are estimates because the script skips those measurements. The README benchmark table also differs from the raw JSON and should be aligned before submission.

### Problem and system design

Single-machine fairness pipelines commonly materialize data in memory. This project explores whether mitigation and evaluation operations can run over Spark DataFrames. Spark reads and transforms rows in partitions; distributed group aggregations estimate protected-group and label frequencies; Spark MLlib trains classifiers; SQL expressions apply prediction adjustments; and aggregate metrics are written to JSON for plotting and notebook review.

| Stage | Method | Distributed work |
|---|---|---|
| Pre-processing | Reweighing | Group and label counts, row weights, weighted Spark model fitting |
| In-processing | Candidate fair models | Train candidates and select using validation fairness and accuracy |
| Post-processing | Reject option classifier | Tune a decision-boundary band on validation predictions and apply group-aware flips |
| Evaluation | Accuracy, SPD, DIR | Aggregate prediction rates and correctness by protected group |

The repository provides Docker and a shell pipeline, along with scripts for correctness, scalability, and plotting. The results notebook reads the committed JSON and displays tables and figures.

### Data and experiment design

Correctness experiments use UCI Adult (income label, sex grouping) and COMPAS (two-year non-recidivism label, race grouping). Scalability experiments use Folktables ACS records from California and Texas. The downloader combines 2018 one-year ACS samples and creates a binary protected field from the ACSIncome group. Benchmark inputs are sampled for target sizes below the available source and formed by repeating the source dataframe above its available size. Repetition is a pipeline stress test; it duplicates observations and does not create an independent population.

All three sources contain person-level records. They are public, but the course brief also prohibits real personal data. Confirm with the instructor that these datasets meet that restriction. If not, substitute a fully synthetic population and rerun the experiments. The accompanying dataset card records the transformations and limitations.

### Fairness measures

Statistical parity difference (SPD) is the favorable prediction rate for the unprivileged group minus that for the privileged group; zero indicates equal rates. Disparate impact ratio (DIR) divides the unprivileged favorable rate by the privileged favorable rate; one indicates equal rates. Accuracy reports overall correct predictions on the test split. These are descriptive measures for specified groups and labels. They do not capture every form of harm, establish legal compliance, or replace domain review. The 0.80 ratio sometimes called the four-fifths rule is a screening convention, not a standalone legal conclusion.

### Correctness results

Values below are copied from `output/correctness_results.json`. Outcomes vary by dataset and mitigation method; no method improves every metric in every dataset.

| Dataset and method | Accuracy | SPD | DIR |
|---|---:|---:|---:|
| Adult baseline | 0.8454 | -0.1670 | 0.3203 |
| Adult reweighing | 0.8409 | -0.0787 | 0.6090 |
| Adult in-processing | 0.8269 | -0.0122 | 0.9090 |
| Adult reject option | 0.8306 | -0.0382 | 0.7479 |
| COMPAS baseline | 0.6706 | -0.2682 | 0.6446 |
| COMPAS reweighing | 0.6680 | -0.2637 | 0.6506 |
| COMPAS in-processing | 0.6263 | -0.1725 | 0.8168 |
| COMPAS reject option | 0.6641 | -0.0317 | 0.9503 |

On Adult, in-processing has the smallest absolute SPD and highest DIR among the listed methods, with lower accuracy than baseline. On COMPAS, reject-option processing has the smallest absolute SPD and highest DIR, while in-processing has the lowest accuracy. These results describe this project's splits, definitions, and settings; they do not imply general fairness guarantees.

### Scalability evidence

The committed raw JSON reports these Spark runtimes. They depend on machine configuration and run conditions. Throughput is also available in the JSON.

| Rows | Reweighing (s) | In-processing (s) | Reject option (s) | Laptop baseline |
|---:|---:|---:|---:|---|
| 50,000 | 12.286 | 31.606 | 8.058 | Measured: 0.634 s; 7.75 MB traced |
| 100,000 | 6.715 | 24.937 | 6.377 | Measured: 3.192 s; 13.92 MB traced |
| 250,000 | 6.805 | 24.957 | 7.102 | Measured: 5.587 s; 34.82 MB traced |
| 500,000 | 7.611 | 27.587 | 7.796 | Skipped; estimated OOM entry |
| 1,000,000 | 8.382 | 30.683 | 9.028 | Skipped; estimated OOM entry |

Spark timings are not monotonic at smaller sizes, consistent with startup, scheduling, caching, and measurement noise. The benchmark script bypasses the laptop baseline above 250,000 rows and writes estimated memory/OOM labels. The README table and notebook narrative now reflect the raw JSON values. The scalability plot was regenerated from the same JSON before submission.

### Limitations and recommendations

- Resolve the course data policy for public person-level records; use fully synthetic data if required.
- Rerun from a clean environment and record hardware, Spark configuration, executor and partition counts, source versions, and run date.
- Label skipped laptop cases as estimates, or measure them under an explicit memory cap.
- Describe row replication as a compute stress test and avoid presenting it as evidence of fairness generalization to larger populations.
- Keep train, validation, and test data separate for tuning and final evaluation. Report group counts and uncertainty where feasible.
- Treat fairness metrics as diagnostic evidence and document harms and trade-offs relevant to any intended deployment.

### Reproducibility and references

Repository: https://github.com/AtharvaGore/bigdataproject. Run instructions and dependencies are in `README.md`; raw result artifacts are in `output/`. The project dataset brief identifies Folktables, COMPAS, and UCI Adult as topic 18 suggestions.

Dataset references: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), [ProPublica COMPAS analysis](https://github.com/propublica/compas-analysis), [Folktables](https://github.com/zykls/folktables).

Method references: Kamiran and Calders, “Data Preprocessing Techniques for Classification Without Discrimination,” *Knowledge and Information Systems* (2012); Agarwal et al., “A Reductions Approach to Fair Classification,” ICML (2018); Kamiran, Karim, and Zhang, “Decision Theory for Discrimination-Aware Classification,” ICDM (2012).
