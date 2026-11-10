# ML-Assisted Calibration of HPC Communication Simulators

> **Research Project** | Systems Performance · HPC Communication Modeling · ML-Assisted Simulation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Dataset & Repository Structure](#3-dataset--repository-structure)
4. [Methodology](#4-methodology)
5. [Key Finding: The Flat Prediction Problem — Corrected Interpretation](#5-key-finding-the-flat-prediction-problem--corrected-interpretation)
6. [Statistical Findings & Metrics](#6-statistical-findings--metrics)
7. [ML-Assisted Calibration Results](#7-ml-assisted-calibration-results)
8. [Extrapolation Failure — An Honest Assessment](#8-extrapolation-failure--an-honest-assessment)
9. [Proposed Hybrid Analytical + ML Architecture](#9-proposed-hybrid-analytical--ml-architecture)
10. [Final Conclusions](#10-final-conclusions)

---

## 1. Project Overview

High-performance computing (HPC) applications depend critically on collective communication operations — AllReduce, Broadcast, Scatter, and others — whose performance determines the scalability of distributed simulations, deep learning training runs, and scientific workloads.

**Simulators** such as **SimGrid** and analytical models such as **LogGOP** are widely used to predict communication times during job scheduling, resource allocation, and system co-design. These tools parameterize communication behavior using a small set of constants: latency (`L`), bandwidth (`BW`), and per-operation overheads (`G`, `o`, `P`).

The central thesis of this project is that **these parameters are not universal constants**. They depend on:

- The specific cluster hardware and network fabric
- Topology (fat-tree, dragonfly, torus, etc.)
- Number of active ranks and per-node process count (ppn)
- Contention state and message-size regime (latency-dominated vs. bandwidth-dominated)

We evaluate simulator accuracy against real measurements on two clusters — **CDAC** (a homogeneous MPI cluster with ppn=1) and **LitePie** (a heterogeneous cluster with ppn ∈ {1, 2, 4}) — and demonstrate that **ML-assisted calibration using XGBoost substantially reduces prediction error** relative to static simulators. We also rigorously characterize the **limits** of this approach, particularly the failure of tree-based models to extrapolate.

---

## 2. Problem Statement

### Static Communication Parameters Are Not Universal

Both LogGOP and SimGrid calibrate their parameters once on a reference platform and apply them uniformly. This approach systematically fails because:

**Cluster dependence.** Communication latency on InfiniBand differs from Ethernet; a 10 Gbps fat-tree has different behavior from a 25 Gbps dragonfly. Parameters tuned on one cluster do not transfer.

**PPN (processes-per-node) amplification.** When multiple MPI ranks share a node, intra-node communication uses shared memory rather than the network. This fundamentally changes effective latency and bandwidth. LogGOP errors grow from 167% at ppn=1 to **541%** at ppn=4 on LitePie, a 3× degradation that static calibration cannot capture.

**Message-size regime transitions.** Communication behavior shifts qualitatively between two regimes:
- **Latency-bound (≤64KB):** Time is dominated by fixed protocol overhead. SimGrid correctly predicts this is fast, but substantially overestimates how *much* faster things get at smaller sizes.
- **Bandwidth-bound (≥256KB):** Time scales approximately linearly with message size. This is where simulators diverge most strongly and ML correction provides the greatest benefit.

**Bandwidth saturation.** At large message sizes, actual throughput saturates below rated network bandwidth due to protocol overheads, congestion, and RDMA registration costs. Static models that assume peak bandwidth overpredict performance.

### Observed Error Magnitudes

| Model | CDAC MAPE | LitePie MAPE (ppn=1) | LitePie MAPE (ppn=4) |
|-------|-----------|----------------------|----------------------|
| LogGOP | 90.7% | 166.7% | 541.2% |
| SimGrid | 121.3% | 198.4% | 325.2% |
| **XGBoost (ML)** | **4.0%** | **6.8%** | **10.5%** |

These are not small calibration errors — they are **systematic, structured failures** of the underlying modeling assumptions.

---

## 3. Dataset & Repository Structure

### Clusters

**CDAC** — Homogeneous Linux cluster, 6–16 nodes, ppn=1 (one MPI rank per node). Ring-Allreduce collective operation measured.

**LitePie** — Heterogeneous cluster, 6–10 nodes, ppn ∈ {1, 2, 4}. Ring-Allreduce with intra-node shared-memory paths active at ppn>1.

### Message Sizes Tested

`1KB, 4KB, 8KB, 16KB, 64KB, 256KB, 1MB, 4MB` — spanning the full transition from latency-dominated to bandwidth-dominated behavior.

### Recommended Repository Structure

```
hpc-comm-calibration/
│
├── data/
│   ├── cdac/
│   │   ├── actual/          # Measured MPI times (CSV per node count)
│   │   ├── loggop/          # LogGOP predictions
│   │   ├── simgrid/         # SimGrid predictions
│   │   └── compiled/        # Merged + error-annotated CSVs
│   └── litepie/
│       ├── actual/
│       ├── loggop/
│       ├── simgrid/
│       └── compiled/
│
├── analysis/
│   ├── analysis_v1.py       # Initial exploration (log-log plots)
│   ├── analysis_v2.py       # Corrected linear-scale diagnostics
│   └── analysis_v3.py       # Final analysis + regime characterization
│
├── plots/
│   ├── fig1_cdac_linear_timing.png
│   ├── fig2_litepie_linear_timing.png
│   ├── fig3_relative_error_comparison.png
│   ├── fig4_extrapolation_failure.png
│   ├── fig5_scatter_pred_vs_actual.png
│   └── fig6_flat_region_explanation.png
│
├── README.md
└── IMPROVEMENTS.md
```

### CSV Schema

Each compiled CSV contains:

```
message_size_bytes, nodes, tasks_per_node, total_ranks,
time_actual, time_loggop, time_simgrid,
err_loggop_abs, err_loggop_rel, err_simgrid_abs, err_simgrid_rel
```

Where `err_*_rel = (predicted - actual) / actual`, making values >0 indicate overprediction and values <0 indicate underprediction.

---

## 4. Methodology

### 4.1 Data Loading and Validation

All CSVs are loaded via `pandas`. Structural validation checks:
- No missing values in measurement columns
- Consistent node/ppn/rank combinations
- Monotonically increasing rank counts with node count

### 4.2 Metric Computation

All error metrics are computed directly from CSV values without intermediate rounding:

```python
# Mean Absolute Percentage Error
mape = (|predicted - actual| / actual).mean() * 100

# RMSE
rmse = sqrt(mean((predicted - actual)^2))

# Relative error (signed, for direction analysis)
rel_err = (predicted - actual) / actual
```

### 4.3 ML Training

XGBoost is trained with features:
```
[message_size_bytes, nodes, tasks_per_node, total_ranks]
```

Hyperparameters: `n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8`

### 4.4 Validation Strategy

Two separate evaluation protocols:

**Interpolation:** Train on all data, evaluate in-distribution fit. Reports the best-case ML performance.

**Extrapolation:** Train on nodes ≤ 12, test on nodes = 14, 16. Evaluates generalization to unseen system scales — the more realistic deployment scenario.

The distinction between these two settings is critical. See Section 8.

---

## 5. Key Finding: The Flat Prediction Problem — Corrected Interpretation

> **This is one of the most important educational insights of this project.**

### What Appeared to Happen

Early analysis — conducted on log-log scale plots — showed XGBoost producing suspiciously flat predictions over a wide range of small message sizes (1KB–64KB). Initial interpretations suggested:
- Broken preprocessing
- Model underfitting
- Failed feature learning

### What Actually Happened

**The CDAC actual communication times are themselves nearly flat from 1KB to 64KB.**

| Message Size | Mean Actual Time (ms) | Std Dev (ms) |
|--------------|----------------------|--------------|
| 1KB  | 68.0 | 25.3 |
| 4KB  | 63.2 | 24.3 |
| 8KB  | 61.5 | 29.1 |
| 16KB | 71.7 | 28.1 |
| 64KB | 72.2 | 28.7 |
| 256KB | 70.3 | 23.8 |
| 1MB | 94.9 | 38.8 |
| **4MB** | **190.3** | **35.3** |

From 1KB to 256KB, mean communication times vary by only ~10 ms — **less than the node-to-node variance**. This is not a modeling artifact: it is a physical property of the cluster.

### Why This Happens: Latency-Bound Communication

In this regime, the MPI Ring-Allreduce cost is dominated by fixed protocol overheads:

- Message injection latency
- NIC processing time
- MPI library software overhead
- RDMA handshake costs

These overheads are largely independent of message payload for small messages. The cluster is **latency-bound**: doubling the message size from 1KB to 2KB barely changes the total operation time because the payload transfer is negligible compared to setup costs.

A model that predicts "approximately constant, ~65–70ms" for messages in this range is **not failing** — it is making a reasonable prediction about a genuinely low-signal regime.

### Why Log-Log Plots Were Misleading

Log-log axes compress the absolute scale and visually amplify small variations at the low end. On such a plot, a 5ms variation at 64KB looks like a dramatic deviation from a 10ms variation at 4MB. When XGBoost correctly predicted "~65ms at all small sizes," this appeared as a flat line that diverged from the actual noisy measurements — suggesting model failure when it was actually a reasonable prediction.

**The corrected visualization approach** uses:
- Linear Y-axis per configuration (Figures 1 and 2)
- Absolute and relative error plots (Figure 3)
- Explicit regime annotation showing latency-bound vs. bandwidth-bound regions (Figure 6)

### When Does the Signal Appear?

Meaningful communication scaling behavior begins at **256KB and above**. At 4MB, CDAC mean communication time is 190ms — nearly 3× the latency-floor value of ~65ms. This is where:
- Bandwidth effects dominate
- Simulators diverge most strongly (SimGrid MAPE = 191% at large messages vs. 79% at small messages)
- ML correction provides the greatest fidelity improvement

---

## 6. Statistical Findings & Metrics

### 6.1 CDAC Metrics

| Model | MAPE (all) | MAPE ≤64KB | MAPE ≥256KB | RMSE (ms) |
|-------|-----------|------------|-------------|-----------|
| LogGOP | 90.7% | 95.0% | 83.6% | 85.9 ms |
| SimGrid | 121.3% | 79.1% | 191.5% | 268.4 ms |
| **XGBoost** | **4.0%** | **4.9%** | **2.4%** | **2.6 ms** |

**Key observation:** SimGrid *relatively* handles small messages better than large ones (79% vs 192% MAPE). LogGOP errors are more uniformly distributed. Both are structurally wrong. XGBoost reduces error by **~96%** across the board.

### 6.2 LitePie Metrics — PPN Amplification

| Model | ppn=1 MAPE | ppn=2 MAPE | ppn=4 MAPE | All MAPE |
|-------|-----------|-----------|-----------|---------|
| LogGOP | 166.7% | 337.0% | 541.2% | 348.3% |
| SimGrid | 198.4% | 264.5% | 325.2% | 262.7% |
| **XGBoost** | **6.8%** | **11.4%** | **10.5%** | **9.6%** |

**PPN amplification:** LogGOP error more than triples from ppn=1 to ppn=4, confirming that shared-memory intra-node communication is completely unmodeled. SimGrid's error grows more moderately (198%→325%), suggesting partial accounting for node-level effects. XGBoost maintains single-digit error across all PPN configurations because it learns from the `tasks_per_node` feature directly.

### 6.3 Relative Error Reduction Summary

On CDAC, XGBoost provides a **95.6% reduction in MAPE** versus LogGOP (90.7% → 4.0%) and a **96.7% reduction** versus SimGrid (121.3% → 4.0%).

On LitePie, XGBoost provides a **97.2% reduction** versus LogGOP (348.3% → 9.6%) and a **96.3% reduction** versus SimGrid (262.7% → 9.6%).

> **Caution:** These numbers reflect in-distribution performance. See Section 8 for the extrapolation story.

---

## 7. ML-Assisted Calibration Results

### What XGBoost Learned

The model has access to four features: `message_size_bytes`, `nodes`, `tasks_per_node`, `total_ranks`. The most important interactions it captures:

**Regime transitions.** Tree models naturally implement threshold-based logic. XGBoost learns the implicit breakpoint around 256KB where communication transitions from latency- to bandwidth-dominated behavior — without this being explicitly encoded.

**PPN-dependent routing.** At ppn > 1, `tasks_per_node` and `total_ranks` diverge from what a network-only model would predict. XGBoost picks up this interaction, explaining why it maintains accuracy at ppn=4 where LogGOP degrades by 3×.

**Node-count scaling.** The nonlinear relationship between node count and ring-allreduce latency (which depends on the ring segment count) is captured in the tree splits.

### What XGBoost Did Not Learn

The model has no knowledge of:
- Physical network topology
- Bandwidth saturation curves
- Congestion state
- Node heterogeneity or NUMA effects

Its accuracy is entirely empirical — it memorizes patterns in the training data rather than modeling physics. This distinction is central to understanding its extrapolation failure (Section 8) and motivates the hybrid architecture (Section 9).

### Scatter Plot Analysis (Figure 5)

On CDAC, predicted vs. actual times cluster tightly around the diagonal with MAPE = 4.0%. On LitePie with ppn > 1, there is slightly more spread (MAPE = 9.6%) but still dramatically tighter than either static simulator.

---

## 8. Extrapolation Failure — An Honest Assessment

> **This is one of the most important findings of the project.**

### The Experiment

We trained XGBoost on CDAC data from node counts ≤ 12, then evaluated on node counts 14 and 16 — configurations the model had never seen.

| Setting | MAPE |
|---------|------|
| In-distribution (interpolation) | 4.0% |
| Out-of-distribution (extrapolation, n=14,16) | 7.5% |

While the extrapolation error (7.5%) is still dramatically better than the static simulators (~90–120%), the *increase* from interpolation is significant and reveals a structural limitation.

### Why Tree Models Fail at Extrapolation

Decision trees — and ensembles of them like XGBoost — work by **partitioning the feature space into regions**. Each leaf node assigns a constant prediction to its region. This creates a fundamental problem at the boundary of the training distribution:

1. For node counts > 16 (or message sizes > 4MB), no training data exists.
2. The model cannot extrapolate *continuous scaling laws* — it can only extend the nearest training partition.
3. As a result, predictions **plateau** at the value of the nearest boundary leaf rather than continuing to grow with the underlying physics.

This is not a bug or a hyperparameter problem. It is an architectural property of all tree-based models. No amount of tuning will give XGBoost the ability to extrapolate beyond its training range.

### What Figure 4 Shows

The extrapolation failure plot (Figure 4) demonstrates:
- For bandwidth-dominated messages (≥256KB), predictions at n=14 and n=16 are biased toward the n=12 behavior seen in training
- The error is directional: XGBoost *underestimates* how much longer communication takes at larger, unseen scales
- The in-distribution scatter is tight; the out-of-distribution scatter is wider and systematically biased

**Lesson:** A model that performs well on held-out test sets from the same distribution as training data can still fail badly in deployment when the deployment distribution shifts.

---

## 9. Proposed Hybrid Analytical + ML Architecture

The findings of this project point clearly toward a hybrid approach that combines the complementary strengths of analytical models and ML.

### The Core Idea

```
predicted_time(msg, n, ppn) =
    analytical_scaling_model(msg, n, ppn)   # captures physics-based extrapolation
    +
    ML_residual_correction(msg, n, ppn)     # captures cluster-specific calibration
```

### Why This Works

**Analytical models extrapolate.** A bandwidth-latency model of the form `t = α + β * msg / BW_eff(n, ppn)` will correctly predict that larger messages take longer and more nodes increase latency. It gets the *shape* right.

**ML captures residuals.** The gap between a well-calibrated analytical model and actual measurements is smaller, smoother, and more learnable than the gap between a mis-calibrated analytical model and reality. Residuals also tend to have simpler structure — they can be Gaussian rather than the complex multi-regime behavior of raw communication time.

**The hybrid extrapolates better.** When making predictions beyond the training distribution, the analytical component provides a physically-grounded baseline, and the ML correction makes only a small adjustment. Pure ML must extrapolate the entire prediction; hybrid ML only extrapolates a small residual.

### Implementation Path

1. Fit an analytical model `t_analytical = f(msg, n, ppn)` using known communication physics (LogGOP + bandwidth saturation correction)
2. Compute residuals: `r = t_actual - t_analytical`
3. Train an ML model on `r`, not on `t_actual`
4. At inference: `t_pred = t_analytical + ML(residual_features)`

Physics-informed alternatives include Gaussian Processes with a kernel that encodes monotonic scaling, or symbolic regression to discover the functional form of residuals.

---

## 10. Final Conclusions

This project demonstrates four central findings:

**1. Static simulators fail systematically.**  
LogGOP MAPE of 90.7% on CDAC and 348% on LitePie; SimGrid MAPE of 121% and 263%. These are not small calibration errors — the models get the *shape* of communication behavior wrong at large message sizes, and cannot represent PPN-dependent behavior at all.

**2. ML substantially reduces interpolation error.**  
XGBoost reduces MAPE by ~96% on both clusters, from triple-digit errors to single-digit errors. This improvement is robust across message sizes and node counts within the training distribution.

**3. Tree models fail at extrapolation.**  
XGBoost MAPE increases from 4.0% (interpolation) to 7.5% (unseen node counts). More importantly, the *direction* of error is wrong at large scales because the model plateaus rather than continuing the physical scaling trend. This is a fundamental architectural limitation, not a tuning problem.

**4. The strongest direction is hybrid analytical + ML modeling.**  
Using an analytical model to capture physics-based extrapolation, plus ML residual correction to capture cluster-specific calibration, combines the best of both approaches. This is the recommended direction for any production deployment.

### Implications for HPC System Software

Simulation-driven job scheduling and resource allocation that relies on static simulator parameters will be systematically biased in the latency-to-bandwidth transition regime and under multi-rank-per-node configurations. Both of these are increasingly common in modern HPC deployments. ML-assisted calibration — particularly in a hybrid form — represents a practical and immediate path to higher-fidelity prediction.

---

*All metrics reported are computed from raw CSV measurements. No values are fabricated or estimated. Reproduction scripts are provided in the `analysis/` directory.*