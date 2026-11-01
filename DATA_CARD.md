# Dataset card

## Project

Distributed Bias Mitigation on Apache Spark

## Data used

The repository reports three datasets:

| Dataset | Reported use | Record type | Attributes used or described |
|---|---|---|---|
| UCI Adult / Census Income | Correctness and fairness comparison | Person-level census records | Sex as the protected attribute; income above $50K as the favorable label |
| COMPAS two-year recidivism | Correctness and fairness comparison | Person-level criminal justice records | Race as the protected attribute; no rearrest within two years as the favorable label |
| Folktables ACS PUMS | Scalability benchmark | Person-level US Census survey records | `protected` is derived from the ACSIncome group (White versus non-White in the downloader); income task label |

The code downloads sources at run time. Dataset versions, retrieval dates, and source checksums should be recorded at the time of the final rerun. The project brief suggests Folktables for this topic and says to verify license and access terms before submission.

## Privacy and permitted use

These are public, person-level records, not synthetic data. Public availability does not make them anonymous or remove the possibility that they contain sensitive information. The course requirement says no real personal data may be used. **Confirm with the instructor that these records are acceptable before submitting.** If the rule excludes person-level public records, replace them with a fully synthetic dataset and rerun all reported experiments. Do not attempt to identify individuals.

## Processing and transformations

- UCI Adult: binary target is income greater than $50K; the downloader encodes male as protected value 1 and female as 0.
- COMPAS: filters to African-American and Caucasian records under the documented screening criteria; the favorable target is no two-year rearrest; Caucasian is encoded as 1 and African-American as 0.
- Folktables: the downloader combines CA and TX 2018 one-year ACS samples and assigns `protected` from the ACSIncome group. Benchmark sizes below the source size are sampled; sizes above it are made by repeating the source dataframe. This repetition creates duplicate records and must be disclosed as a runtime stress test rather than independent population growth.

## Known limitations

- Sensitive group definitions are simplified binary comparisons and omit other identities and intersections.
- Results are dataset-, label-, split-, model-, and threshold-dependent; they do not establish that a system is fair or legally compliant.
- The checked-in benchmark JSON says single-node runs were successful through 250,000 rows. At 500,000 and 1,000,000 rows, it records an OOM status with memory estimates; the benchmark script skips those runs rather than demonstrating an observed crash.
- The JSON result values differ from the table currently printed in README. Use the committed JSON as the source of reported raw values until a fresh run replaces it.
- ACS samples contain people represented by survey records; row replication does not create new people or independent observations.

## Sources

- UCI Adult: https://archive.ics.uci.edu/dataset/2/adult
- ProPublica COMPAS analysis data: https://github.com/propublica/compas-analysis
- Folktables project: https://github.com/zykls/folktables
- Course dataset brief: Project_Datasets_Level3_Set1.docx, topic 18.

