# Cyber Attack Detection on Smart Grids

Intrusion detection and analysis pipeline for IEC 60870-5-104 smart-grid traffic using feature engineering, model benchmarking, and adversarial robustness checks.

## Paper Context

This work is related to the IEEE reference: [https://ieeexplore.ieee.org/document/11083563](https://ieeexplore.ieee.org/document/11083563)

## Project Scope

This repository focuses on end-to-end cyberattack analysis for smart-grid communications:

- Traffic and feature analysis from IEC-104 network captures.
- Supervised attack detection with multiple model families.
- Ensemble and ablation experiments.
- Explainability and robustness-oriented evaluations.

## Repository Layout

```text
.
|- README.md
|- ids.py / ids1.py / honey*.py (security and analysis scripts)
|- results/ (automated analysis outputs)
|- IEEE_Plots/ (visualization for publications)
|- Graphs/ (exploratory data analysis)
|- docs/ (architecture and screenshots)
```

## Architecture

The pipeline consists of the following stages:
1. **Data Acquisition**: IEC-104 traffic and logs.
2. **Preprocessing**: Data cleaning and normalization.
3. **Feature Engineering**: Selecting and transforming relevant features.
4. **Model Training**: Benchmarking Decision Trees, Random Forests, SVM, and KNN.
5. **Evaluation**: Generating confusion matrices, metrics, and ablation studies.

## Problem Statement

Smart-grid communication protocols are vulnerable to cyberattacks that can disrupt critical infrastructure. The goal is to detect malicious traffic patterns reliably while handling large-scale and noisy data.

## Benchmark Results

| Model | Accuracy | Notes |
| --- | --- | --- |
| Decision Tree | 91.16% | Fast training and inference |
| Random Forest | 91.66% | Robust performance baseline |
| SVM | 76.65% | Sensitive to high dimensionality |
| KNN | 76.65% | Memory-intensive evaluation |
| Ensemble | 91.81% | Optimal overall performance |

## How To Run

1. Prepare the dataset paths in the relevant script.
2. Run preprocessing and feature analysis.
3. Train baseline models using the `ids.py` or `ids1.py` scripts.
4. Execute evaluation and ablation scripts to generate results.

## Future Work

- Integration of a unified orchestration pipeline.
- Enhanced configuration for training/validation/test splits.
- Implementation of advanced anomaly-focused metrics.
