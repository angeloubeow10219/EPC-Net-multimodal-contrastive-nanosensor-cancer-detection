# Multimodal contrastive learning fuses electrochemical and plasmonic nanosensor signals for early cancer

EPC-Net aligns voltammetric electrochemical signals and SERS or SPR spectra in a 512-dimensional joint space. The model combines asymmetric encoders, bidirectional InfoNCE, a mutual-information-gap penalty, cross-modal reconstruction, supervised classification, batch-aware sampling, and domain-randomized signal perturbations.

## Scope

The release contains the computational analytical layer. It does not claim that molecular bridge cohorts are paired electrochemical and plasmonic measurements. TCGA and AACR Project GENIE provide molecular bridge data, while the BMC Medicine study provides the serum-SERS literature anchor. A study-specific raw SERS accession was not stated in the manuscript and is therefore not invented here.

## Layout

- `code/epcnet/models`: electrochemical convolution-state-space encoder, plasmonic Transformer-state-space encoder, projection, classification, and reconstruction heads
- `code/epcnet/losses`: bidirectional InfoNCE, modality-distance term, marginal KL term, reconstruction loss, and four-term objective
- `code/epcnet/data`: paired HDF5 storage, batch-aware sampling, normalization, and instrument perturbations
- `code/epcnet/signalops`: reusable electrochemical and spectral processing operators
- `code/epcnet/training`: distributed execution, cosine scheduling, mixed precision, and atomic state persistence
- `code/epcnet/evaluation`: fixed-specificity metrics, bootstrap intervals, calibration, concordance, and random-effects analysis
- `configs`: primary experiment and component ablations
- `datasets.txt`: verified official or canonical data and study links

## Installation

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### conda

```bash
conda env create -f environment.yml
conda activate epcnet
pip install -e .
```

### Docker

```bash
docker build -t epcnet .
```

## Data

The preparation command accepts two aligned sample-by-feature matrices and one metadata table.

```bash
epc-prepare \
  --electrochemical data/ec.csv \
  --plasmonic data/sers.csv \
  --metadata data/metadata.csv \
  --output data/paired.h5
```

The metadata file must contain `label`, `cohort`, `instrument`, and `acquisition_date` integer columns. Matrix rows must have identical sample ordering.

Official sources and their access terms are listed in `datasets.txt`. TCGA expression and clinical metadata are available through the GDC portal. GENIE v18.0-public is available through Synapse and cBioPortal under the provider terms. The BMC article is CC BY 4.0; its article identifier is included as a provenance anchor, not presented as a raw spectral archive.

## Training

The reported primary setup uses four NVIDIA A100 GPUs, batch size 256 paired samples, 100 epochs, AdamW, peak learning rate `1e-4`, weight decay `0.05`, cosine decay, and approximately 72 hours.

```bash
torchrun --standalone --nproc-per-node=4 \
  -m epcnet.cli.train \
  --config configs/main.yaml \
  --data data/paired.h5 \
  --seed 42
```

The five manuscript seeds are `42`, `123`, `314`, `271`, and `1234`. The effective global batch is 256 because the configuration batch size denotes the global stratified batch divided across four processes.

## Objective

The total loss is

```text
L = 1.0 Lcontrast + 0.5 Lrecon + 0.3 Lsupervised + 0.2 LMI-gap
```

The primary values are temperature `0.10`, marginal KL coefficient `0.025`, and modality-dropout probability `0.30`. The electrochemical encoder has four convolution-state-space layers. The plasmonic encoder uses twelve interleaved selective state-space and attention layers.

## Evaluation

The evaluation package supports sensitivity at fixed 99% specificity, specificity, AUROC, F1, PPV, NPV, Brier score, calibration error, 10,000-resample confidence intervals, independent-cohort differences, Benjamini-Hochberg correction, per-site MCID crossing, Cohen kappa, and DerSimonian-Laird random-effects aggregation.

The manuscript reports the following pooled reference values:

| Measure | Reported value |
|---|---:|
| Sensitivity at 99% specificity | 65.0% |
| Bootstrap 95% interval | 60.5–69.5% |
| Specificity | 99.6% |
| AUROC | 0.962 |
| Cross-site sensitivity spread | 5.5 percentage points |
| Throughput | approximately 220 samples per second on one A100 |
| Model footprint | approximately 80 million parameters |
| Per-sample compute | approximately 12 GFLOPs |

Exact numerical agreement requires the study-specific paired tensors and cohort splits. The public molecular portals alone cannot reconstruct electrochemical measurements that were not deposited.

## Experiment variants

The configuration directory includes loss-removal, temperature, modality-dropout, and encoder-capacity-ratio experiments. Every variant inherits the main configuration and changes only the named factor.

## Compute budget

The reported training budget is four A100 GPUs for approximately 72 hours. The manuscript does not state the A100 memory variant, storage footprint, gradient accumulation, or exact warmup duration. The release uses bfloat16, no gradient accumulation, and a five-epoch warmup as explicit operational defaults. These values should not be interpreted as reported measurements.

## License

Apache License 2.0.

