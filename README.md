## Can Open-Source LLMs Predict Scientific Impact? A Reproducibility Study of LLM-Based Impact Prediction

This repository contains code, configuration files, and precomputed analysis outputs for reproducing the experiments described in our reproducibility paper. The goal of this repository is to enable end‑to‑end replication of the main results.

**Paper Abstract:** This paper reports on a reproducibility study of LLM-based scientific impact prediction framed as an IR evaluation problem. We systematically vary four experimental factors—target variable choice, LLM-based measurement stability, normalization robustness, and learning objective—to determine which findings generalize and which depend on specific implementation choices.

---

## Key Findings

Our reproducibility study reveals four main insights:

### RQ1: Cross-Target Alignment and Generalization
- **Novelty & Engagement dominates**: The LLM-derived "Novelty & Engagement" component consistently correlates most strongly with citation-based impact across GPT-4o, DeepSeek-V3, and Llama-3.3-70B
- **Bounded generalization**: The pattern replicates under domain shift from biomedical (PLOS ONE) to Computer Science (arXiv, ICLR) but fails for lower-capability models (Llama-3.1-8B)
- **Normalization persistence**: The component ordering remains stable across raw citations, TNCSI, and TNCSI_SP, though effect sizes attenuate under normalization
- **Peer review alignment**: Novelty & Engagement also correlates with contemporaneous peer review scores (ρ=0.161 for GPT-4o on NAIDv2)

### RQ2: Measurement Sensitivity
- **Item ordering matters**: Interleaved presentation of positive/negative attributes collapses semantic coherence (antonym consistency → 0) even for GPT-4o
- **Polarity-grouped orderings preserve structure**: Positive-first and negative-first orderings maintain strong negative antonym consistency (ρ < -0.7) for capable models
- **Component-specific failures**: Llama-3.1-8B exhibits semantic inversion on the Novelty & Engagement component (positive antonym consistency), explaining its RQ1 alignment failures
- **Discrimination range separates models**: GPT-4o shows Δρ=0.64 between most/least informative items; Llama-3.1-8B shows only Δρ=0.19

### RQ3: Sensitivity of Normalized Metrics
- **Keyword quality is non-monotonic**: DeepSeek-V3 achieves best NED (0.259) while GPT-4o performs worst (0.334) on TKPD dataset
- **Systematic score inflation**: Open-source LLMs produce higher TNCSI scores than GPT-3.5-Turbo baseline (Δ≈+0.17, Cohen's d≈0.68)
- **TNCSI_SP is more robust**: ICC(2,1)=0.930 (excellent) for TNCSI_SP vs 0.878 (good) for TNCSI when comparing OSS models on NAIDv2
- **Cross-generator agreement**: Open-source models (DeepSeek, Qwen, Llama-3.3) show good mutual consistency, but including GPT-3.5 degrades agreement

### RQ4: Learning Objective Sensitivity
- **Dataset maturity matters**: Pointwise MSE dominates on established papers (NAIDv1), but pairwise BCE achieves superior NDCG@20 (+0.064) on recent papers (NAIDv2)
- **Contradicts prior conclusions**: Original NAIDv1 paper claimed MSE uniformly superior; we show pairwise ranking excels when citation targets are noisy
- **Target choice is critical**: TNCSI_SP yields NDCG@20=0.797 on NAIDv2 vs 0.217 for raw citations, demonstrating normalization is essential for sparse citation settings
- **Objective-evaluation alignment**: Regression optimizes MAE; ranking optimizes top-k utility. Choice should match downstream use case.

---
## Tables

### RQ1: Cross-Target Alignment (All Models; Spearman ρ)

**NAIDv1**

| Model | Target | A&U | N&E | Q&R | Replicated (N&E strongest) |
|---|---|---:|---:|---:|:---:|
| GPT-4o | Raw citations | 0.028 | **0.255** | 0.202 | ✅ |
| GPT-4o | TNCSI | 0.048 | **0.177** | 0.168 | ✅ |
| GPT-4o | TNCSI_SP | 0.032 | **0.207** | 0.199 | ✅ |
| DeepSeek-V3 | Raw citations | 0.139 | **0.256** | -0.015 | ✅ |
| DeepSeek-V3 | TNCSI | 0.131 | **0.170** | 0.012 | ✅ |
| DeepSeek-V3 | TNCSI_SP | 0.149 | **0.221** | 0.008 | ✅ |
| Llama-3.3-70B | Raw citations | 0.063 | **0.241** | 0.074 | ✅ |
| Llama-3.3-70B | TNCSI | 0.089 | **0.156** | 0.048 | ✅ |
| Llama-3.3-70B | TNCSI_SP | 0.079 | **0.191** | 0.065 | ✅ |
| Llama-3.1-8B | Raw citations | **0.085** | -0.057 | -0.009 | ❌ |
| Llama-3.1-8B | TNCSI | **0.087** | -0.030 | 0.006 | ❌ |
| Llama-3.1-8B | TNCSI_SP | **0.123** | -0.017 | 0.038 | ❌ |

**NAIDv2**

| Model | Target | A&U | N&E | Q&R | Replicated (N&E strongest) |
|---|---|---:|---:|---:|:---:|
| GPT-4o | Raw citations | 0.125 | **0.165** | 0.088 | ✅ |
| GPT-4o | TNCSI | 0.095 | **0.165** | 0.080 | ✅ |
| GPT-4o | TNCSI_SP | 0.094 | **0.174** | 0.079 | ✅ |
| DeepSeek-V3 | Raw citations | **0.128** | 0.094 | 0.004 | ❌ |
| DeepSeek-V3 | TNCSI | **0.104** | 0.102 | 0.029 | ❌ |
| DeepSeek-V3 | TNCSI_SP | 0.109 | **0.126** | 0.049 | ✅ |
| Llama-3.3-70B | Raw citations | 0.054 | **0.101** | 0.040 | ✅ |
| Llama-3.3-70B | TNCSI | 0.042 | **0.104** | 0.035 | ✅ |
| Llama-3.3-70B | TNCSI_SP | 0.035 | **0.110** | 0.033 | ✅ |
| Llama-3.1-8B | Raw citations | -0.066 | **0.034** | -0.072 | ~0 |
| Llama-3.1-8B | TNCSI | -0.033 | **0.020** | -0.055 | ~0 |
| Llama-3.1-8B | TNCSI_SP | **0.036** | -0.024 | -0.001 | ~0 |

**Peer review alignment (NAIDv2; Spearman ρ between N&E and peer review signals)**

| Model | score_mean | acceptance |
|---|---:|---:|
| GPT-4o | 0.161 | 0.137 |
| DeepSeek-V3 | 0.097 | 0.079 |
| Llama-3.3-70B | 0.085 | 0.089 |
| Llama-3.1-8B | 0.026 | -0.010 |


---

### RQ2: Measurement Stability (Mean Antonym Consistency, averaged over 3 components)

Values closer to **-1.0** indicate stronger semantic polarity (better instrument coherence).

| Model | NAIDv1 (Positive-first) | NAIDv2 (Positive-first) | NAIDv1 (Interleaved) | NAIDv2 (Interleaved) |
|---|---:|---:|---:|---:|
| GPT-4o | -0.829 | -0.866 | 0.022 | 0.021 |
| DeepSeek-V3 | -0.884 | -0.923 | -0.090 | -0.119 |
| Llama-3.3-70B | -0.681 | -0.758 | -0.026 | -0.090 |
| Llama-3.1-8B | -0.451 | -0.462 | -0.198 | -0.178 |

---

### RQ3: Keyword Generator Sensitivity

**Keyword quality on TKPD (mean NED ↓):**

| Model | Mean NED ↓ |
|---|---:|
| **DeepSeek-V3** | **0.259** |
| **Llama-3.3-70B** | **0.273** |
| **Qwen-2.5-72B** | **0.284** |
| Llama-3.1-8B | 0.296 |
| GPT-3.5-Turbo | 0.301 |
| Qwen-2.5-7B | 0.321 |
| GPT-4o | 0.334 |

**Cross-generator agreement (ICC(2,1); higher = more consistent):**

| Dataset | Metric | Scope | ICC(2,1) |
|---|---|---|---:|
| NAIDv2 | TNCSI_SP | OSS-only (3 models) | **0.930** |
| NAIDv2 | TNCSI | OSS-only (3 models) | 0.878 |
| NAIDv1 | TNCSI_SP | OSS-only (3 models) | 0.898 |
| NAIDv1 | TNCSI | OSS-only (3 models) | 0.849 |
| NAIDv1 | TNCSI_SP | All-4 (incl. GPT-3.5) | 0.809 |
| NAIDv1 | TNCSI | All-4 (incl. GPT-3.5) | 0.686 |

---

### RQ4: Learning Objective Sensitivity (TNCSI_SP prediction; Llama-3.1-8B)

| Dataset | Objective | MAE ↓ | NDCG@20 ↑ | Spearman ρ ↑ |
|---|---|---:|---:|---:|
| NAIDv1 | Pointwise (MSE) | **0.209** | **0.890** | **0.489** |
| NAIDv1 | Pairwise (BCE) | 0.337 | 0.885 | 0.455 |
| NAIDv2 | Pointwise (MSE) | 0.280 | 0.797 | **0.358** |
| NAIDv2 | Pairwise (BCE) | **0.263** | **0.861** | 0.326 |

**Target choice matters (Pointwise MSE; Llama-3.1-8B):**

| Dataset | Target | MAE ↓ | NDCG@20 ↑ | Spearman ρ ↑ |
|---|---|---:|---:|---:|
| NAIDv2 | TNCSI_SP | 0.280 | **0.797** | 0.358 |
| NAIDv2 | TNCSI | **0.175** | 0.549 | **0.376** |
| NAIDv2 | Raw citations | 18.537 | 0.217 | 0.214 |
| NAIDv1 | TNCSI_SP | **0.209** | **0.890** | 0.489 |
| NAIDv1 | TNCSI | 0.163 | 0.607 | 0.484 |
| NAIDv1 | Raw citations | 25.036 | 0.614 | **0.623** |

---


## Repository Structure

At the top level:

```text
scientific-impact/
├── naid/
│   ├── v1_resource/          # Code and data for NAIDv1 experiments
│   └── v2_resource/          # Code and data for NAIDv2 experiments
├── llm_judge/                # Scripts to derive impact metrics from LLM outputs
├── analysis_results/         # Precomputed tables and correlation/loadings CSVs
├── data/                     # Aggregated JSON metrics for main results
├── .gitignore
└── README.md
```

Key subdirectories:

- **`naid/v1_resource`**:  
  Code and resources for NAIDv1 (e.g., `NAIDv1/dataset.py`, previous methods, TKPD keyword experiments, and v1 training scripts).
- **`naid/v2_resource`**:  
  Code and resources for NAIDv2, including fine‑tuning (`v2_finetune.py`), simple training scripts, and run directories such as `runs/0212_0112_simple_TNCSI_SP_20k/`.
- **`llm_judge/`**:  
  Scripts such as `get_paper_metrics.py` and `sensitivity_report.py` that compute and analyze impact‑related metrics from LLM outputs.
- **`analysis_results/`**:  
  Versioned CSVs (e.g., factor loadings, correlations, readability components) used to build the paper's tables and figures.
- **`data/`**:  
  Aggregated JSON files (e.g., `naidv1_test_aggregated_metrics.json`, `naidv2_test_aggregated_metrics.json`) summarizing performance.

---

## Datasets

This repository assumes access to the following datasets:

- **NAIDv1**  
  Collection of ~12K Computer Science papers from arXiv with:
  - Raw citation counts (frozen snapshot)
  - TNCSI and TNCSI_SP scores  
  - Additional paper metadata (e.g., in `naid/v1_resource/NAIDv1/NAIDv1_*_extrainfo.csv`)

- **NAIDv2**  
  Collection of ~24K peer‑reviewed ICLR submissions (2021-2025) with:
  - Raw citation counts (frozen snapshot)
  - Reviewer scores (score_mean) and acceptance decisions
  - Variant files with and without citation fields (e.g., `NAIDv2-train-with-cites.csv`)

- **TKPD (Topic Keyword Prediction Dataset)**  
  Dataset used to evaluate keyword generation quality and its effect on normalized citation metrics.  
  Relevant files live under `naid/v1_resource/TKPD/` (e.g., `TKPD.csv`, `ned_output_*/`).

---

## Core Pipelines

Below we describe the main components of the experimental pipeline and how they connect to this repository.

### 1. Data Loading and Preprocessing

- **NAIDv1**:  
  Implemented in `naid/v1_resource/NAIDv1/dataset.py`, which loads train/test CSVs and associated metadata.  
  Large CSVs such as `NAIDv1_train_extrainfo.csv` and `NAIDv1_test_extrainfo.csv` are expected to be present but may not be redistributed here.

- **NAIDv2**:  
  Implemented in `naid/v2_resource/NAIDv2/dataset.py`, which reads `NAIDv2-*.csv` splits (with and without citation information).

### 2. LLM-Based Abstract Assessment (RQ1, RQ2)

The 60-item structured assessment instrument is implemented in `llm_judge/`:

**Key Implementation Details:**
- **Deterministic decoding**: All LLMs use temperature=0 to ensure reproducibility
- **Rating scale**: Each attribute rated 0-100 (no chain-of-thought prompting)
- **Models evaluated**: GPT-4o, DeepSeek-V3, Llama-3.3-70B-Instruct, Llama-3.1-8B-Instruct
- **Orderings tested**: positive-first, negative-first, interleaved (30 positive + 30 antonym attributes)
- **PCA extraction**: Varimax rotation applied to extract 3 latent components

**Antonym Consistency Check:**
For each component, compute correlation between loadings of attribute pairs:
```
antonym_consistency = corr(loading[positive_attr], loading[antonym_attr])
```
Values < -0.7 indicate strong semantic coherence; values near 0 or positive indicate measurement failure.

**Running the assessment:**
```bash
python llm_judge/get_paper_metrics.py \
  --model gpt-4o \
  --dataset naidv1 \
  --ordering positive_first
```

### 3. Keyword Generation and TNCSI/TNCSI_SP Computation (RQ3)

The directory `naid/v1_resource/TKPD/` contains:

- `TKPD.csv` – the topic keyword prediction dataset with reference keywords
- `prompt_keyword_async_search.py` – script(s) for generating keywords with different LLMs
- `ned_output_*/` – LLM‑specific outputs (e.g., `ned_output_gpt4o/`, `ned_output_llama3.3/`, `ned_output_deepseek/`) with `ned_results_*.csv` and summary JSONs

**TNCSI Computation:**
1. Generate topic keyword from title+abstract using LLM
2. Retrieve semantically similar papers from Semantic Scholar using keyword query
3. Fit exponential distribution to citation counts of retrieved cohort: CDF(c) = 1 - exp(-(c - loc)/scale)
4. Compute percentile score for target paper

**TNCSI_SP Computation:**
Same as TNCSI, but filter retrieved cohort to papers within ±6 months of target publication date

**Keyword Quality Evaluation:**
```bash
python naid/v1_resource/TKPD/prompt_keyword_async_search.py \
  --model deepseek-v3 \
  --output_dir ned_output_deepseek
```

Normalized Edit Distance (NED) computed as:
```
NED = edit_distance(generated, reference) / max(len(generated), len(reference))
```

### 4. Supervised Training on NAIDv2 (RQ4)

The main fine‑tuning entry point is:

```bash
python naid/v2_resource/v2_finetune.py \
  --config naid/v2_resource/accelerate_config.yaml \
  --train_csv naid/v2_resource/NAIDv2/NAIDv2-train-with-cites.csv \
  --test_csv naid/v2_resource/NAIDv2/NAIDv2-test-with-cites.csv \
  --target tncsi_sp \
  --loss_type mse  # or 'pairwise_bce'
```

**Training Configuration:**
- **Backbone**: Llama-3.1-8B-Instruct (shared across all objectives)
- **Optimization**: LoRA (Low-Rank Adaptation) for memory efficiency
- **Objectives**:
  - Pointwise: MSE loss on numeric target
  - Pairwise: Binary cross-entropy on preference labels I[target_a > target_b]
- **Targets**: raw_citations, tncsi, tncsi_sp
- **Evaluation**: MAE (numeric accuracy), NDCG@20 (ranking quality), Spearman ρ (rank correlation)

**Pairwise Training Details:**
Pairs constructed within dataset cohorts. At inference, model produces pointwise scores directly (sigmoid-normalized for pairwise-trained models).

Run outputs (metrics, predictions, tokenizer and adapter configs) are saved under `naid/v2_resource/runs/...`.

For lighter‑weight baselines or sanity checks, you can also use:

- `naid/v2_resource/simple_train.py` – a simplified training script on NAIDv2.

### 5. Sensitivity Analysis and Reporting

The `llm_judge/` directory includes scripts such as:

- `get_paper_metrics.py` – computes LLM‑based metrics for each paper
- `sensitivity_report.py` – aggregates and reports sensitivity analyses across models and conditions

The resulting CSVs and summaries are stored in `analysis_results/` and `data/`, and map directly to reported tables and figures.

---

## Reproducing the Main Results

Because many large intermediate files are already provided under `analysis_results/` and `data/`, you can reproduce the paper's tables in two ways:

### Option 1: Using Precomputed Results (Fast, No GPU Required)

- Load the CSVs under `analysis_results/ver*/` and the JSON summaries in `data/`
- These files correspond to:
  - **Table 2**: `analysis_results/ver*/main_correlations.csv`
  - **Table 3**: `analysis_results/ver*/antonym_consistency.csv`
  - **Table 4**: `analysis_results/ver*/antonym_consistency_detail.csv`
  - **Table 6**: `data/tncsi_sensitivity_results.json`
  - **Tables 7-8**: `data/naidv1_test_aggregated_metrics.json`, `data/naidv2_test_aggregated_metrics.json`

### Option 2: Full Re-Run (GPU Recommended)

1. **Place datasets** under `naid/v1_resource/` and `naid/v2_resource/` in the expected file layout
2. **Run RQ1/RQ2 experiments** (LLM-based assessment):
   ```bash
   python llm_judge/get_paper_metrics.py --model gpt-4o --dataset naidv1
   python llm_judge/get_paper_metrics.py --model deepseek-v3 --dataset naidv1
   python llm_judge/get_paper_metrics.py --model llama-3.3-70b --dataset naidv1
   ```
3. **Run RQ3 experiments** (keyword generation):
   ```bash
   python naid/v1_resource/TKPD/prompt_keyword_async_search.py --model deepseek-v3
   python naid/v1_resource/TKPD/prompt_keyword_async_search.py --model qwen-2.5-72b
   ```
4. **Run RQ4 experiments** (supervised training):
   ```bash
   # Pointwise regression
   python naid/v2_resource/v2_finetune.py --target tncsi_sp --loss_type mse
   
   # Pairwise ranking
   python naid/v2_resource/v2_finetune.py --target tncsi_sp --loss_type pairwise_bce
   
   # Compare targets
   python naid/v2_resource/v2_finetune.py --target raw_citations --loss_type mse
   python naid/v2_resource/v2_finetune.py --target tncsi --loss_type mse
   ```
5. **Aggregate results**:
   ```bash
   python llm_judge/sensitivity_report.py --output_dir analysis_results/
   ```
6. Compare generated tables to precomputed CSVs in `analysis_results/`

Exact command‑line arguments and hyperparameters are documented within the corresponding scripts (via `argparse` help messages and inline comments).

---

## Reproducibility Checklist

To ensure reproducibility of your own experiments:

- [ ] **Report LLM models** with exact version strings (e.g., `gpt-4o-2024-08-06`, `deepseek-v3`)
- [ ] **Fix decoding** to temperature=0 for deterministic outputs
- [ ] **Specify item ordering** for multi-attribute assessment (positive-first recommended)
- [ ] **Verify semantic coherence** using antonym consistency before interpreting PCA components
- [ ] **Document keyword generator** used for TNCSI/TNCSI_SP computation
- [ ] **Report citation snapshot date** to ensure targets are frozen and comparable
- [ ] **Align evaluation metrics** with learning objective (MAE for regression, NDCG@k for ranking)
- [ ] **Control for dataset maturity** when comparing pointwise vs pairwise objectives

---

