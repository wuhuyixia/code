# Reproducible Decentralized Federated Learning Experiments

This repository hosts the experimental code associated with the manuscript revision. It is organized around a configuration-driven simulator so that datasets, client partitions, models, gradient estimators, Byzantine attacks, aggregators, communication topologies, random seeds, logs, and metrics can be reproduced from explicit settings.

## Repository layout

```text
.
├── main.py
├── config/default.yaml
├── core/
├── data/
├── models/
├── estimator/
├── attack/
├── agg/
├── network/
├── utils/
├── scripts/
├── data_cache/
├── results/
└── run_logs/
```

## Reproducibility policy

Every run resolves the YAML configuration, records the random seed and device information, and writes metrics and a resolved configuration under the run directory. The final manuscript-specific hyperparameters and exact implementations of paper-specific components will be added only after they are checked against the experiments reported in the manuscript.

## Status

Initial framework scaffold. Standard components are being wired first; paper-specific `Algorithm 1` and OPRF details are intentionally not inferred from the manuscript without verification.
