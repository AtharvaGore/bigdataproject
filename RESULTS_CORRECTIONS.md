# Results corrections before submission

The README scalability table and results notebook have been updated to match the committed `output/scalability_results.json`. Current raw JSON values are:

| Rows | Spark reweighing (s) | Spark in-processing (s) | Spark post-processing (s) | Laptop baseline |
|---:|---:|---:|---:|---|
| 50,000 | 12.286 | 31.606 | 8.058 | measured success; 0.634 s; 7.75 MB traced peak |
| 100,000 | 6.715 | 24.937 | 6.377 | measured success; 3.192 s; 13.92 MB traced peak |
| 250,000 | 6.805 | 24.957 | 7.102 | measured success; 5.587 s; 34.82 MB traced peak |
| 500,000 | 7.611 | 27.587 | 7.796 | **not run**; JSON records estimated 400 MB and an OOM status |
| 1,000,000 | 8.382 | 30.683 | 9.028 | **not run**; JSON records estimated 800 MB and an OOM status |

These values are the contents of the committed raw JSON, not a fresh rerun. The script intentionally bypasses the laptop baseline above 250,000 rows, so call those entries estimates or rerun under a declared memory cap and report the observed outcome. The checked-in scalability PNG predates these corrections; regenerate it with `python plot_results.py` before presenting the plot.

The input scaling procedure samples the dataset for smaller sizes and repeats the source dataframe for larger sizes. State this clearly in the report and plots. Repeated rows can characterize pipeline execution cost but cannot support a claim that fairness outcomes generalize to a larger independent population.

Also update README claims that imply the 80% rule proves legal fairness or that the repository has an MIT license unless a matching `LICENSE` file is added.

