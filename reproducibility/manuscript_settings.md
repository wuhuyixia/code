# Manuscript-reported experimental settings

This file records only settings that are explicitly stated in the current
manuscript. It is a provenance aid, **not a substitute for the exact executable
experiment YAML files**. Values not stated in the manuscript must be recovered
from the original experiment configurations or run logs rather than guessed.

## Common settings

- Clients: 12
- Regular clients: 10
- Byzantine clients: 2
- Communication rounds: 100
- Client partition: Dirichlet
- Dirichlet parameter beta: 0.5
- Batch size: 64
- Sign-flipping amplification factor gamma: 5
- Validation split: 10% stratified split of the official training set
- Loss: cross-entropy
- Main-comparison seeds: 42, 43, 44
- Component-ablation seeds: 42--51

## MNIST

- Classes: 10
- Input channels: 1
- Local epochs: 1
- Learning rate: 0.05
- Optimizer: SGD
- Gaussian standard deviation: 3.00
- OPRF perturbation magnitude delta: 0.00100

## Fashion-MNIST

- Classes: 10
- Input channels: 1
- Local epochs: 1
- Learning rate: 0.01
- Optimizer: SGD with momentum
- Gaussian standard deviation: 1.40
- OPRF perturbation magnitude delta: 0.00071

## GTSRB

- Classes: 43
- Input channels: 3
- Local epochs: 2
- Learning rate: 0.01
- Optimizer: SGD with momentum
- Gaussian standard deviation: 0.85
- OPRF perturbation magnitude delta: 0.00034
- Communication rounds: 100
- Topology switching interval: 20 rounds
- Snapshot activation rounds: 0, 20, 40, 60, 80

## Model architecture stated in the manuscript

The classification model uses two convolutional layers with 32 and 64 output
channels. Each is followed by ReLU and 2x2 max pooling. The extracted features
are processed by adaptive average pooling, a fully connected layer with 128
hidden units and ReLU, and a final classification layer. The output dimension is
10 for MNIST/Fashion-MNIST and 43 for GTSRB.

## Reported execution environment

- Operating system: Windows 10
- GPU: NVIDIA GeForce RTX 3090 (24 GB)
- Python: 3.8.13
- PyTorch: 2.4.0
- CUDA: 12.1
- cuDNN: 9.1.0

## Values that still require recovery from original logs/configurations

Before archival release, the exact executable configurations should additionally
record, where applicable:

- Byzantine client IDs for each reported run;
- exact static topology / GTSRB adjacency snapshots;
- momentum coefficient and weight decay;
- exact image preprocessing/resize/normalization settings;
- estimator initialization/smoothing options;
- aggregation-specific settings;
- any dataset paths or dataset-version identifiers;
- exact versions of Python dependencies not explicitly reported in the paper.

These values must be copied from the original experiment records. They should
not be inferred from defaults or invented for the archival repository.
