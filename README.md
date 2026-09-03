# Reproducible Decentralized Federated Learning Experiments

This repository provides the configuration-driven experimental framework associated with the manuscript revision. The framework separates reusable code from manuscript-specific experimental values: `config/default.yaml` defines the available fields but intentionally does **not** encode the final paper hyperparameters.

## Repository layout

```text
.
├── main.py
├── config/
│   └── default.yaml
├── core/
│   ├── simulator.py
│   ├── client.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── metrics.py
├── data/
│   ├── datasets.py
│   └── partition.py
├── models/
│   ├── cnn.py
│   ├── mlp.py
│   └── factory.py
├── estimator/
│   ├── base.py
│   ├── autograd.py
│   ├── zeroth_order.py
│   ├── oprf.py
│   └── factory.py
├── attack/
│   ├── none_attack.py
│   ├── sign_flip.py
│   ├── gaussian.py
│   └── factory.py
├── agg/
│   ├── dfedavg.py
│   ├── krum.py
│   ├── trimmed_mean.py
│   ├── algorithm1.py
│   └── factory.py
├── network/
│   └── topology.py
├── utils/
├── scripts/
│   ├── run_seeds.py
│   ├── summarize_results.py
│   └── environment_info.py
└── requirements.txt
```

## Framework behavior

The simulator uses synchronous communication rounds. In each round all clients first perform local optimization, configured Byzantine transformations are applied to outgoing messages, every receiver aggregates only the models visible under the current topology, and all receiver models are then updated simultaneously. This prevents client execution order from introducing asynchronous effects.

Supported framework components include:

- IID and Dirichlet client partitioning;
- MNIST, Fashion-MNIST, CIFAR-10 and local CSV-layout GTSRB loaders;
- configurable CNN/MLP model construction;
- PyTorch autograd, symmetric two-point zeroth-order estimation and OPRF;
- no attack, sign-flipping and Gaussian perturbation hooks;
- D-FedAvg, Krum, Trimmed Mean and manuscript-aligned Algorithm 1;
- complete or deterministic connected dynamic k-neighbor communication graphs;
- per-client and per-round metrics, oracle/forward/backward counts, message count, transmitted payload bytes and runtime measurements.

## Installation

```bash
python -m pip install -r requirements.txt
```

Record the software/hardware environment with:

```bash
python scripts/environment_info.py
```

## Configuration

Copy `config/default.yaml` to an experiment-specific YAML file and fill every parameter required by that experiment. Fields set to `null` are intentionally unresolved. The framework raises an error rather than silently substituting paper-specific hyperparameters.

Example invocation structure:

```bash
python main.py --config config/experiments/example.yaml
```

Individual settings can be overridden without editing the YAML:

```bash
python main.py --config config/experiments/example.yaml --set experiment.seed=<seed>
```

## Multiple independent runs

Use the multi-seed runner after creating a complete experiment configuration:

```bash
python scripts/run_seeds.py --config config/experiments/example.yaml --seeds <seed1> <seed2> <seed3>
```

Final-round mean and sample standard deviation can then be summarized from the generated run directories:

```bash
python scripts/summarize_results.py run_logs/<run1> run_logs/<run2> run_logs/<run3>
```

## Run artifacts

Each run writes a resolved configuration and reproducibility artifacts under its run directory:

- `resolved_config.yaml`: complete resolved run configuration;
- `partition.json`: sample indices and class counts assigned to every client;
- `topologies/*.json`: communication graph whenever the topology changes;
- `client_metrics.csv`: local training/evaluation and estimator counters;
- `round_metrics.csv`: aggregate accuracy, estimator counts, communication volume and timing;
- `summary.json`: completed rounds and total elapsed time.

Dataset files, generated logs, checkpoints and results are intentionally excluded from Git tracking by `.gitignore`.

## Manuscript-specific configurations

The reusable framework is now separated from the final manuscript settings. Before archival release, experiment-specific YAML files will be added for the exact dataset preprocessing, model architecture parameters, optimizer settings, attack parameters, topology settings and seeds used to generate each reported figure/table. No placeholder value in `default.yaml` should be interpreted as a reported experimental setting.
