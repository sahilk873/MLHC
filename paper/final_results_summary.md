# Final Results Summary

- Core regression results: see `results/metrics/final_summary.json`.
- External validation: ENCoDE all-pairs (617) with skin-tone bins populated.
- Fairness gaps and cluster-robust CIs: see `reports/encode_cluster_ci.md`.
- Conformal coverage: see `reports/conformal_results.md`.
- Mondrian conformal: see `reports/conformal_mondrian.md`.
- Worst-group safety: see `reports/worst_group_safety.md`.
- Occult hypoxemia classifier: see `reports/occult_hypoxemia_classifier.md`.

## Threats to validity
- Repeated measures may inflate effective sample size; we mitigate with cluster bootstrap.
- Occult hypoxemia prevalence is low and unstable across subgroups; classifier metrics are sensitive to this.