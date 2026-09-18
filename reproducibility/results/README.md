# Seed-level manuscript results

This directory is reserved for the **seed-level numerical results used in the manuscript tables and figures**.

For the main MNIST and Fashion-MNIST comparisons, the manuscript reports matched
runs for seeds 42, 43, and 44. Store the final test accuracies used for each
paired comparison in a wide CSV with one row per seed, for example:

```csv
seed,Algorithm1,DFedAvg,Krum,TrimmedMean
42,<value>,<value>,<value>,<value>
43,<value>,<value>,<value>,<value>
44,<value>,<value>,<value>,<value>
```

Do not reconstruct or infer missing values from means, standard deviations, or
figures. The committed values must come from the original run logs.

The component ablation uses ten matched seeds (42--51). Archive those seed-level
values separately from the three-seed main comparison.

Once the original seed-level CSV is present, reproduce the manuscript statistics
with:

```bash
python scripts/statistical_analysis.py \
  --input reproducibility/results/<dataset>.csv \
  --reference Algorithm1 \
  --comparators DFedAvg Krum TrimmedMean \
  --output reproducibility/results/<dataset>_statistics.csv
```

The script reports the paired mean difference, sample standard deviation of the
paired differences, Student-t confidence interval, two-sided Wilcoxon
signed-rank p-value, Holm-adjusted p-value across the supplied comparisons, and
paired rank-biserial effect size.
