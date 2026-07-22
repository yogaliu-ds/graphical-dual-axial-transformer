# Graphical Dual Axial Transformer for Risk-Adjusted Portfolio Optimization

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"Graphical Dual Axial Transformer for Risk-Adjusted Portfolio Optimization"**.

This repository provides a PyTorch research framework that combines a **Graphical Lasso**-derived asset relationship graph with a **dual-axis Transformer** to learn risk-adjusted, end-to-end differentiable portfolio allocation strategies directly from multi-asset time-series data.

## Overview

Traditional portfolio optimization separates return/risk forecasting from weight allocation. This framework instead trains a single model end-to-end to map historical price data directly to portfolio weights, optimized against risk-adjusted objectives (e.g., Sharpe ratio, mean-variance utility, Sortino ratio, CVaR).

Two ideas drive the model design:

- **Graph-guided attention** — A sparse asset-relationship graph is estimated from historical returns via **Graphical Lasso** and injected as an attention mask into the cross-sectional (asset) axis of the Transformer, so the model attends more strongly to structurally related assets.
- **Dual-axis attention** — Asset relationships and temporal dynamics are modeled by two separate attention axes rather than being flattened into one sequence:
    1. **N-axis (cross-sectional)**: relationships between assets at the same time step, optionally masked by the Graphical Lasso adjacency matrix.
    2. **T-axis (temporal)**: each asset's own time-series dynamics, with a causal mask to prevent information leakage from future time steps.

Model outputs (asset scores) are converted into portfolio weights via a pluggable allocation strategy (long-only softmax, long-short, or generalized softmax) and evaluated with a comprehensive suite of portfolio performance metrics.

## Key Features

- **Graph-guided cross-sectional attention** using Graphical Lasso–estimated adjacency matrices (full / positive-only / negative-only correlation graphs).
- **Graphical Dual Axial Transformer (GDAT)** architecture with independent N-axis (asset, graph-masked) and T-axis (temporal, causally masked) attention.
- **Interchangeable portfolio construction strategies**: long-only softmax, constrained long-short (abs-max), and generalized softmax.
- **Risk-aware loss functions**: Sharpe ratio and mean-variance utility, each available in both covariance-based and historical-simulation variants, plus historical-simulation-only Sortino ratio and CVaR objectives.
- **Extensive backtest evaluation**: annualized return, standard deviation (with linear shrinkage), Sharpe/Sortino ratio, maximum drawdown, turnover, VaR/CVaR, and stress-test drawdown.
- **Configuration-driven grid search** over datasets, model hyperparameters, seeds, and loss functions.

## Model Architecture

```
INPUT PIPELINE:
Financial Data -> Feature Engineering -> Tensor Preparation -> Model Forward Pass

[CSV Files] -> [Returns / Range] -> [B, N, T, E] -> [Axial Transformer] -> [Portfolio Weights] -> [Portfolio Metrics]
     |                |                    |                  |                    |                     |
Price/High/Low   simple returns,     Batched sliding      N-axis: graph      Softmax / AbsMax /     Sharpe / Return /
   data          high-low range      time-series windows  T-axis: causal     GeneralizedSoftmax      Risk metrics
                                                            attention masking
```

*For clarity, this diagram shows only the primary `x` path through the model. As described below, each sample also carries a longer raw-return history (`x_cov`) that bypasses the Transformer entirely and feeds directly into risk calculations.*

**Data flow:**

1. **Sample extraction** (`extract_time_series_batches` in `scripts/train.py`) — for every prediction date, a sliding window over the dataset produces three tensors plus a date label:
   - `x`: `[B, N, T, E]` — the recent `sequence_length`-day feature window (default `T=21`; `E=2` features: simple returns and high-low range returns). This is the only tensor fed into the score model.
   - `y`: `[B, N, horizon]` — realized future returns (default `horizon=1`), used to score realized performance in the loss and evaluation metrics.
   - `x_cov`: `[B, N, burn_in_period]` — a longer raw-return history (default `burn_in_period=252` days) ending at the same point as `x`. It is **not** passed through the Transformer; it is used only by covariance-based and historical-simulation risk terms in the loss functions and evaluation metrics.

2. **Score prediction** (`models/score_block.py`, `models/axial_transformer.py`) — the `AxialTransformer` processes `x` (optionally masked by the graph adjacency matrix) through separate N-axis and T-axis attention, producing one score per asset representing the model's expectation of future performance.

3. **Weight generation** (`models/portfolio_block.py`) — asset scores are converted into portfolio weights via a chosen strategy:
   - `PortfolioBlockSoftmax`: long-only, weights sum to 1.
   - `PortfolioBlockAbsMax`: long-short, `sum(|w_i|) = 1` with a per-asset bound.
   - `PortfolioBlockGeneralizedSoftmax`: alternative normalization scheme.

4. **Loss computation** (`utils/loss.py`) — portfolio weights are combined with `y` (return) and/or `x_cov` (risk) into a risk-adjusted loss (see [Loss Functions](#loss-functions) below).

5. **Evaluation** (`utils/evaluation_metrics.py`) — a held-out test set is backtested using `y` and `x_cov` together, and scored across a broad set of return, risk, and turnover metrics (see [Evaluation Metrics](#evaluation-metrics)).

### Score Block

The score model is `ScoreBlockGDAT` in `models/score_block.py` (config `score_block` key: `'GDAT'`) — a thin wrapper around the `AxialTransformer` described above, producing one score per asset from the dual-axis attention output.

### Loss Functions

| Name (`loss_type` key) | Class | Description |
|---|---|---|
| `Cov_SharpeRatio` | `Cov_SharpeRatioLoss` | Sharpe ratio using a shrinkage covariance matrix. |
| `Cov_MeanVariance` | `Cov_MeanVarianceLoss` | Mean-variance utility using a shrinkage covariance matrix. |
| `HistoricalVariance_SharpeRatio` | `HistoricalVariance_SharpeRatioLoss` | Sharpe ratio via historical return simulation. |
| `HistoricalVariance_MeanVariance` | `HistoricalVariance_MeanVarianceLoss` | Mean-variance utility via historical return simulation. |
| `HistoricalVariance_SortinoRatio` | `HistoricalVariance_SortinoRatioLoss` | Sortino ratio (downside risk) via historical simulation. |
| `HistoricalVariance_CVaR` | `HistoricalVariance_CVaRLoss` | Conditional Value at Risk optimization. |
| `ExpectedReturn` | `ExpectedReturnLoss` | Pure return maximization with an L2 penalty on weights. |

### Evaluation Metrics

`utils/evaluation_metrics.py` provides `PortfolioMetrics`, organized by calculation basis:

- **Realized-return metrics**: Expected Return, Maximum Drawdown, Positive Ratio (% of periods with positive returns), Turnover.
- **Covariance-based metrics**: Std, Sharpe, and Mean-Variance Utility, each in shrinkage and non-shrinkage variants, plus a shrinkage-only Sortino approximation.
- **Historical-simulation metrics**: Std, Sharpe, Sortino, Mean-Variance Utility, all computed from simulated historical portfolio returns.
- **Tail-risk metrics**: VaR (95%), CVaR (95%), Expected Shortfall Ratio, stress-test Maximum Drawdown (computed over the combined `x_cov` + `y` history).

## Repository Structure

```
.
├── config.py               # Central configuration: dataset paths, split dates, model/training hyperparameters
│
├── models/
│   ├── axial_transformer.py  # Core dual-axis (N-axis / T-axis) attention Transformer
│   ├── score_block.py        # GDAT score-generating model (wraps AxialTransformer)
│   └── portfolio_block.py    # Score-to-weight allocation strategies
│
├── utils/
│   ├── common.py              # Data loading, seeding, experiment logging utilities
│   ├── loss.py                 # Risk-adjusted portfolio loss functions
│   └── evaluation_metrics.py   # Backtest evaluation metrics
│
├── scripts/
│   ├── train.py               # Main entry point: data pipeline, training loop, grid search, evaluation
│   └── graphical_lasso.py      # Estimates the asset adjacency graph from historical returns (run before training)
│
├── data/                  # NOT included — user-provided datasets (see Data Preparation below)
└── experiments/            # Output directory for logs and result CSVs (auto-created at runtime)
```

## Installation

Developed and tested with Python 3.13 and PyTorch 2.6.0 (CUDA 12.6 build).

```bash
git clone <repository-url>
cd R26-package
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.6.0` without a CUDA suffix, since CUDA-specific wheels aren't available on PyPI directly. To install the exact CUDA 12.6 build used for development:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

For other CUDA versions or a CPU-only install, use the [PyTorch install selector](https://pytorch.org/get-started/locally/) and then install the remaining dependencies from `requirements.txt`.

## Data Preparation

Datasets are **not included** in this repository and must be supplied by the user under `data/`. Each dataset needs at minimum daily Close, High, and Low price CSVs, indexed by date with asset tickers as columns:

```
data/
└── <DATASET_NAME>/
    ├── <DATASET_NAME>_Close.csv
    ├── <DATASET_NAME>_High.csv
    └── <DATASET_NAME>_Low.csv
```

Register each dataset in the `DATASETS` dictionary in `config.py` (see [Configuration](#configuration) below), specifying file paths, train/validation/test split dates, and Graphical Lasso parameters.

## Usage

### 1. Generate the asset relationship graph

The N-axis attention mask requires a precomputed adjacency matrix per dataset:

```bash
python -m scripts.graphical_lasso
```

This reads each dataset's training-period returns, fits a `GraphicalLasso` model, and writes the resulting adjacency matrices (full, positive-only, negative-only) plus diagnostic plots to `data/<DATASET_NAME>/graphical_lasso/`.

### 2. Configure the experiment

Edit `config.py` to set dataset paths/splits and default model/training hyperparameters, and edit the `GRID_SEARCH_CONFIG` dictionary at the top of `config.py` to define the grid search space (model hyperparameters, learning rates, seeds, loss functions, etc.).

### 3. Run training and evaluation

```bash
python -m scripts.train
```

`scripts/train.py` runs a grid search over the configured parameter space. For each combination, it trains the GDAT model, backtests on the held-out test period, and aggregates results.

### 4. Inspect results

Each run creates a timestamped directory under `experiments/` containing:
- `experiment.log` — full training/evaluation log
- `experiment_results.csv` — one row per grid search configuration, with all evaluation metrics and per-date portfolio returns

## Configuration

All configuration lives in `config.py`:

- **`DATASETS`** — per-dataset file paths, train/validation/test split dates (`T0_date`, `T1_date`, `T2_date`), covariance shrinkage factor, and Graphical Lasso parameters (`alpha`, `max_iter`, `tol`, `mode`).
- **`DEFAULT_DATASET`** — fallback dataset name.
- **`CONFIG['model']`** — model/data hyperparameters: `burn_in_period`, `sequence_length`, `horizon`, `batch_size`, `num_epochs`, `learning_rate`, `risk_aversion`, `risk_free_rate`.
- **`CONFIG['training']`** — `early_stop_patience`, `min_train_loss`, `max_train_samples`, `sharpe_loss_mode`.
- **`GRID_SEARCH_CONFIG`** — the parameter grid used by `scripts/train.py` (GDAT hyperparameters, loss types, datasets, seeds, portfolio strategies, etc.).
- **`PORTFOLIO_METRICS_MODE`** — `"micro"` (average of per-sample ratios) or `"macro"` (ratio of averaged return/risk) aggregation for evaluation metrics.

## Citation

If you use this codebase in your research, please cite:

> Youjia Liu and Yasumasa Matsuda. **Graphical Dual Axial Transformer for Risk-Adjusted Portfolio Optimization.**

## License

This project is licensed under the [MIT License](LICENSE).
# graphical-dual-axial-transformer
# graphical-dual-axial-transformer
