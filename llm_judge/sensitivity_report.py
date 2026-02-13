# import pandas as pd
# import numpy as np
# import json
# from scipy import stats
# from collections import defaultdict
# import itertools

# # ============================================================
# # CONFIGURATION
# # ============================================================
# # JSON structure: { paper_id: { "cites": [...], "TNCSI": [...], "TNCSI_SP": [...] } }
# #
# # Index order in value lists:
# #   NAIDv1: [0]=DeepSeek  [1]=Qwen  [2]=Llama  [3]=GPT-3.5 (from original CSV)
# #   NAIDv2: [0]=DeepSeek  [1]=Qwen  [2]=Llama  (no GPT-3.5 — never computed)
# #
# # Analysis plan:
# #   A) Cross-model consistency among the 3 open-source models — BOTH datasets
# #   B) Comparison vs. GPT-3.5 baseline — NAIDv1 ONLY
# #      (motivates the argument that metrics need to be updated with newer LLMs)

# NAIDV1_PATH = "/mnt/data/son/Reviewerly/dataset/naidv1_test_aggregated_metrics.json"
# NAIDV2_PATH = "/mnt/data/son/Reviewerly/dataset/naidv2_test_aggregated_metrics.json"

# # The 3 open-source challenger models (same indices in both datasets)
# OSS_MODELS  = ["DeepSeek-V3", "Qwen-2.5-72B", "Llama-3.3-70B"]
# OSS_INDICES = [0, 1, 2]

# # GPT-3.5 baseline (NAIDv1 only, appended as index 3)
# BASELINE_LABEL = "GPT-3.5 (baseline)"
# BASELINE_IDX   = 3

# METRICS = ["TNCSI", "TNCSI_SP"]

# # ============================================================
# # HELPERS
# # ============================================================

# def load_data(path: str) -> dict:
#     with open(path, "r") as f:
#         return json.load(f)


# def extract_paired(mapping: dict, metric: str, idx_a: int, idx_b: int):
#     """Two aligned arrays for papers where both indices have valid scores."""
#     a_vals, b_vals = [], []
#     for data in mapping.values():
#         vals = data[metric]
#         if len(vals) > max(idx_a, idx_b):
#             va, vb = vals[idx_a], vals[idx_b]
#             if va != -1 and vb != -1:
#                 a_vals.append(va)
#                 b_vals.append(vb)
#     return np.array(a_vals), np.array(b_vals)


# def extract_subset_valid(mapping: dict, metric: str, indices: list):
#     """
#     Return (paper_ids, matrix) for papers that have valid scores at ALL
#     requested indices. Matrix columns follow the order of `indices`.
#     """
#     ids, rows = [], []
#     for pid, data in mapping.items():
#         vals = data[metric]
#         if len(vals) > max(indices):
#             subset = [vals[i] for i in indices]
#             if all(v != -1 for v in subset):
#                 ids.append(pid)
#                 rows.append(subset)
#     mat = np.array(rows) if rows else np.empty((0, len(indices)))
#     return ids, mat


# def spearman(a, b):
#     if len(a) < 3:
#         return float("nan"), float("nan")
#     return stats.spearmanr(a, b)


# def cohens_d(a, b):
#     diff = a - b
#     return 0.0 if np.std(diff) == 0 else np.mean(diff) / np.std(diff)


# def wilcoxon_test(a, b):
#     diff = a - b
#     if np.all(diff == 0) or len(diff) < 5:
#         return float("nan"), float("nan")
#     try:
#         return stats.wilcoxon(a, b)
#     except Exception:
#         return float("nan"), float("nan")


# def calculate_icc21(data_matrix: np.ndarray) -> float:
#     """ICC(2,1) – two-way random effects, single measures."""
#     n, k = data_matrix.shape
#     if n < 2 or k < 2:
#         return float("nan")
#     grand_mean = np.mean(data_matrix)
#     row_means  = np.mean(data_matrix, axis=1)
#     col_means  = np.mean(data_matrix, axis=0)
#     ms_rows = k * np.sum((row_means - grand_mean) ** 2) / (n - 1)
#     ms_cols = n * np.sum((col_means - grand_mean) ** 2) / (k - 1)
#     ss_res  = np.sum(
#         (data_matrix - row_means[:, None] - col_means[None, :] + grand_mean) ** 2
#     )
#     ms_res = ss_res / ((n - 1) * (k - 1))
#     denom  = ms_rows + (k - 1) * ms_res + k * (ms_cols - ms_res) / n
#     return float("nan") if denom == 0 else (ms_rows - ms_res) / denom


# def cv_per_paper(matrix: np.ndarray) -> np.ndarray:
#     means = np.abs(np.mean(matrix, axis=1))
#     stds  = np.std(matrix, axis=1)
#     with np.errstate(invalid="ignore", divide="ignore"):
#         return np.where(means > 0, stds / means, np.nan)


# def icc_label(icc: float) -> str:
#     if np.isnan(icc): return "N/A"
#     if icc > 0.90:    return "Excellent"
#     if icc > 0.75:    return "Good"
#     if icc > 0.50:    return "Moderate"
#     return "Poor"


# def fisher_ci(rho: float, n: int):
#     """95% CI via Fisher z-transform; clips rho to avoid arctanh(+-1)."""
#     rho_c = np.clip(rho, -0.9999, 0.9999)
#     z     = np.arctanh(rho_c)
#     se    = 1 / np.sqrt(n - 3) if n > 3 else float("nan")
#     return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


# def sig_stars(p):
#     if np.isnan(p): return "n.s."
#     if p < 0.001:   return "***"
#     if p < 0.01:    return "**"
#     if p < 0.05:    return "*"
#     return "n.s."


# def sep(title="", width=72):
#     print(f"\n{'='*width}\n{title}\n{'='*width}" if title else "=" * width)


# def high_variance_papers(mapping, metric, indices, top_n=20):
#     rows = []
#     for pid, data in mapping.items():
#         vals = data[metric]
#         if len(vals) > max(indices):
#             subset = [vals[i] for i in indices]
#             if all(v != -1 for v in subset):
#                 arr = np.array(subset)
#                 rows.append((pid, arr, arr.max() - arr.min()))
#     rows.sort(key=lambda x: x[2], reverse=True)
#     return rows[:top_n]


# # ============================================================
# # PART A — CROSS-MODEL CONSISTENCY (3 OSS models, both datasets)
# # ============================================================

# def part_a_consistency(mapping, dataset_name, metric):
#     """ICC, CV, range, and pairwise correlations for the 3 OSS models only."""

#     # --- Pairwise Spearman ---
#     sep(f"[{dataset_name}] {metric} – Pairwise Spearman Correlation (OSS models)")
#     n = len(OSS_MODELS)
#     corr_mat = np.full((n, n), np.nan)
#     for i, j in itertools.combinations(range(n), 2):
#         a, b = extract_paired(mapping, metric, OSS_INDICES[i], OSS_INDICES[j])
#         if len(a) >= 3:
#             rho, _ = spearman(a, b)
#             corr_mat[i, j] = corr_mat[j, i] = rho
#     np.fill_diagonal(corr_mat, 1.0)

#     print(f"\n  {'':24s}", end="")
#     for name in OSS_MODELS:
#         print(f"{name[:14]:>14s}", end="")
#     print()
#     for i, name in enumerate(OSS_MODELS):
#         print(f"  {name[:24]:24s}", end="")
#         for j in range(n):
#             v = corr_mat[i, j]
#             print(f"{'—':>14s}" if np.isnan(v) else f"{v:>14.4f}", end="")
#         print()

#     # --- ICC + CV + Range ---
#     sep(f"[{dataset_name}] {metric} – Overall OSS Consistency (ICC / CV / Range)")
#     ids, matrix = extract_subset_valid(mapping, metric, OSS_INDICES)
#     n_complete = len(ids)

#     result = {}
#     if n_complete < 10:
#         print(f"  Only {n_complete} papers with complete data — skipping.")
#         return result

#     icc    = calculate_icc21(matrix)
#     cv     = cv_per_paper(matrix)
#     ranges = matrix.max(axis=1) - matrix.min(axis=1)
#     label  = icc_label(icc)

#     print(f"  Models: {', '.join(OSS_MODELS)}")
#     print(f"  Papers with complete data : {n_complete}")
#     print(f"  ICC(2,1)   = {icc:.4f}  [{label}]")
#     print(f"  Mean CV    = {np.nanmean(cv):.4f}  (median {np.nanmedian(cv):.4f})")
#     print(f"  Mean Range = {np.mean(ranges):.4f}  (median {np.median(ranges):.4f})")
#     print(f"  Range > 0.10 : {np.mean(ranges > 0.10)*100:.1f}% of papers")
#     print(f"  Range > 0.20 : {np.mean(ranges > 0.20)*100:.1f}% of papers")

#     # Retrieval divergence
#     sep(f"[{dataset_name}] {metric} – Retrieval Divergence (OSS models, top-20)")
#     top_papers = high_variance_papers(mapping, metric, OSS_INDICES, top_n=20)
#     if top_papers:
#         hdr = "  ".join(f"{m[:8]:>8s}" for m in OSS_MODELS)
#         print(f"\n  {'Paper ID':>36s}  {'Range':>8s}  {'Min':>8s}  {'Max':>8s}  [{hdr}]")
#         print(f"  {'-'*100}")
#         for pid, arr, rng in top_papers:
#             s = "  ".join(f"{v:8.3f}" for v in arr)
#             print(f"  {pid:>36s}  {rng:8.4f}  {arr.min():8.4f}  {arr.max():8.4f}  [{s}]")
#         _, all_mat = extract_subset_valid(mapping, metric, OSS_INDICES)
#         if all_mat.shape[0] > 0:
#             all_ranges = all_mat.max(axis=1) - all_mat.min(axis=1)
#             print()
#             for pct in [50, 75, 90, 95, 99]:
#                 print(f"  P{pct:>2d} range: {np.percentile(all_ranges, pct):.4f}")

#     result = {
#         "dataset": dataset_name, "metric": metric,
#         "scope": "OSS-only (3 models)",
#         "n_complete": n_complete,
#         "ICC": icc, "ICC_label": label,
#         "mean_CV": np.nanmean(cv),
#         "mean_range": np.mean(ranges),
#     }
#     return result


# # ============================================================
# # PART B — COMPARISON vs. GPT-3.5 BASELINE (NAIDv1 only)
# # ============================================================

# def part_b_vs_gpt35(mapping, metric):
#     """
#     Rank correlation + distributional shift for each OSS model vs. GPT-3.5.
#     NAIDv1 only. Used to argue that the original GPT-3.5-based metric
#     is not interchangeable with newer open-source generators.
#     """
#     dataset_name = "NAIDv1"

#     # --- Rank Correlation ---
#     sep(f"[{dataset_name}] {metric} – Rank Correlation vs. {BASELINE_LABEL}")
#     corr_rows = []
#     for oss_idx, model in zip(OSS_INDICES, OSS_MODELS):
#         a_base, b_chal = extract_paired(mapping, metric, BASELINE_IDX, oss_idx)
#         n = len(a_base)
#         if n < 3:
#             print(f"  {model}: insufficient data (n={n})")
#             continue
#         rho, pval = spearman(a_base, b_chal)
#         if abs(rho) == 1.0:
#             print(f"  NOTE: ρ = {rho:.4f} for {model} — perfect correlation, CI degenerate")
#         ci_lo, ci_hi = fisher_ci(rho, n)
#         stars = sig_stars(pval)
#         print(
#             f"  {model:22s}  n={n:5d}  ρ={rho:.4f}  "
#             f"CI=[{ci_lo:.3f},{ci_hi:.3f}]  p={pval:.4e} {stars}"
#         )
#         corr_rows.append({
#             "Model": model, "n": n, "rho": rho,
#             "ci_lo": ci_lo, "ci_hi": ci_hi,
#             "p": pval, "sig": stars,
#         })

#     # --- Distributional Shift ---
#     sep(f"[{dataset_name}] {metric} – Distributional Shift vs. {BASELINE_LABEL}")
#     shift_rows = []
#     for oss_idx, model in zip(OSS_INDICES, OSS_MODELS):
#         a_base, b_chal = extract_paired(mapping, metric, BASELINE_IDX, oss_idx)
#         if len(a_base) < 5:
#             print(f"  {model}: insufficient data (n={len(a_base)})")
#             continue
#         diff           = b_chal - a_base
#         mean_diff      = np.mean(diff)
#         median_diff    = np.median(diff)
#         w_stat, w_pval = wilcoxon_test(a_base, b_chal)
#         d              = cohens_d(b_chal, a_base)
#         pct_higher     = np.mean(b_chal > a_base) * 100
#         pct_lower      = np.mean(b_chal < a_base) * 100
#         pct_equal      = 100 - pct_higher - pct_lower
#         direction      = "more lenient (up)" if mean_diff > 0 else "stricter (down)"
#         stars          = sig_stars(w_pval)

#         print(f"\n  {model}")
#         print(f"    Mean delta={mean_diff:+.4f}  Median delta={median_diff:+.4f}  Cohen's d={d:+.4f}")
#         print(f"    Wilcoxon p={w_pval:.4e} {stars}  ->  {direction}")
#         print(f"    Papers: {pct_higher:.1f}% higher, {pct_lower:.1f}% lower, {pct_equal:.1f}% equal")

#         shift_rows.append({
#             "Model": model, "n": len(a_base),
#             "mean_delta": mean_diff, "median_delta": median_diff,
#             "cohens_d": d, "wilcoxon_p": w_pval, "sig": stars,
#             "direction": direction,
#         })

#     # --- ICC including GPT-3.5 (all 4 models, NAIDv1) ---
#     sep(f"[{dataset_name}] {metric} – Consistency: All 4 Models incl. {BASELINE_LABEL}")
#     all_indices = OSS_INDICES + [BASELINE_IDX]
#     all_labels  = OSS_MODELS  + [BASELINE_LABEL]
#     ids, matrix = extract_subset_valid(mapping, metric, all_indices)
#     n_complete  = len(ids)

#     result_4 = {}
#     if n_complete >= 10:
#         icc    = calculate_icc21(matrix)
#         cv     = cv_per_paper(matrix)
#         ranges = matrix.max(axis=1) - matrix.min(axis=1)
#         label  = icc_label(icc)
#         print(f"  Models: {', '.join(all_labels)}")
#         print(f"  Papers with complete data : {n_complete}")
#         print(f"  ICC(2,1)   = {icc:.4f}  [{label}]")
#         print(f"  Mean CV    = {np.nanmean(cv):.4f}  (median {np.nanmedian(cv):.4f})")
#         print(f"  Mean Range = {np.mean(ranges):.4f}  (median {np.median(ranges):.4f})")
#         print(f"  Range > 0.10 : {np.mean(ranges > 0.10)*100:.1f}% of papers")
#         print(f"  Range > 0.20 : {np.mean(ranges > 0.20)*100:.1f}% of papers")
#         result_4 = {
#             "dataset": dataset_name, "metric": metric,
#             "scope": f"All 4 (incl. {BASELINE_LABEL})",
#             "n_complete": n_complete,
#             "ICC": icc, "ICC_label": label,
#             "mean_CV": np.nanmean(cv),
#             "mean_range": np.mean(ranges),
#         }
#     else:
#         print(f"  Only {n_complete} papers with complete data across all 4 models — skipping.")

#     return corr_rows, shift_rows, result_4


# # ============================================================
# # MAIN
# # ============================================================

# def main():
#     sep("RQ3 - STEP 2: METRIC STABILITY ANALYSIS ACROSS LLM KEYWORD GENERATORS")
#     print(f"  Open-source models : {OSS_MODELS}")
#     print(f"  Baseline (v1 only) : {BASELINE_LABEL} @ index {BASELINE_IDX}")
#     print(f"  Metrics            : {METRICS}")

#     summary_rows = []

#     # ── PART A: OSS cross-consistency on both datasets ─────────────────────
#     sep("PART A: OSS CROSS-MODEL CONSISTENCY  (NAIDv1 & NAIDv2)", width=72)
#     print("  Compares DeepSeek, Qwen, Llama against each other.")
#     print("  NAIDv2 has no GPT-3.5 data -- this is the only valid comparison there.\n")

#     for dataset_name, path in [("NAIDv1", NAIDV1_PATH), ("NAIDv2", NAIDV2_PATH)]:
#         mapping = load_data(path)
#         sep(f"DATASET: {dataset_name}  ({len(mapping)} papers)", width=72)
#         for metric in METRICS:
#             r = part_a_consistency(mapping, dataset_name, metric)
#             if r:
#                 summary_rows.append(r)

#     # ── PART B: NAIDv1 challengers vs. GPT-3.5 ────────────────────────────
#     sep("PART B: OSS MODELS vs. GPT-3.5 BASELINE  (NAIDv1 only)", width=72)
#     print("  Motivates replacing the GPT-3.5 keyword generator:")
#     print("  lower rho and systematic distributional shift show the baseline")
#     print("  metric is not interchangeable with newer open-source generators.\n")

#     mapping_v1 = load_data(NAIDV1_PATH)
#     for metric in METRICS:
#         _, _, result_4 = part_b_vs_gpt35(mapping_v1, metric)
#         if result_4:
#             summary_rows.append(result_4)

#     # ── MASTER SUMMARY TABLE ───────────────────────────────────────────────
#     sep("MASTER SUMMARY TABLE")
#     if summary_rows:
#         df = pd.DataFrame(summary_rows)[
#             ["dataset", "metric", "scope", "n_complete",
#              "ICC", "ICC_label", "mean_CV", "mean_range"]
#         ]
#         df.columns = ["Dataset", "Metric", "Scope", "n",
#                       "ICC(2,1)", "ICC label", "Mean CV", "Mean Range"]
#         print(df.to_string(index=False))

#     sep("INTERPRETATION GUIDE")
#     print("""
# PART A -- OSS Cross-Model Consistency:
#   Answers: "Do DeepSeek / Qwen / Llama agree with each other?"
#   ICC > 0.90  -> The three models are interchangeable for TNCSI computation.
#   ICC 0.75-0.90 -> Good agreement; minor per-paper deviations.
#   ICC < 0.75  -> Metric is sensitive even among open-source generators.

# PART B -- OSS vs. GPT-3.5 Baseline (NAIDv1 only):
#   Answers: "Does the choice of GPT-3.5 vs. newer models matter?"
#   Low rho (< 0.75) + significant Wilcoxon shift  -> strong argument that
#     GPT-3.5-generated TNCSI scores are NOT reproducible with current models.
#   High rho (> 0.90) -> rankings preserved even if absolute values drift.

# Distributional Shift direction:
#   Mean delta > 0 (more lenient) -> challenger generates broader keywords
#     -> larger retrieved cohort -> lower normalization baseline -> inflated TNCSI.
#   Mean delta < 0 (stricter) -> narrower keywords -> smaller cohort
#     -> higher baseline -> deflated TNCSI.
#   Cohen's |d| < 0.2 = negligible; 0.2-0.5 = small; > 0.5 = moderate/large.

# Retrieval Divergence:
#   Range > 0.20 across models indicates the Semantic Scholar API returned
#   qualitatively different cohorts for the same paper -- driven by keyword
#   specificity differences rather than model quality per se.
# """)


# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
import json
from scipy import stats
from collections import defaultdict
import itertools

# ============================================================
# CONFIGURATION
# ============================================================
# Paths to aggregated metric files per dataset and model
# Each JSON has structure: { paper_id: { "cites": [...], "TNCSI": [...], "TNCSI_SP": [...] } }
# Index order within each list must match MODEL_NAMES below.

DATASETS = {
    "NAIDv1": "/mnt/data/son/Reviewerly/dataset/naidv1_test_aggregated_metrics.json",
    "NAIDv2": "/mnt/data/son/Reviewerly/dataset/naidv2_test_aggregated_metrics.json",
}

# Must match the column order stored in the JSON value lists
MODEL_NAMES = ["DeepSeek-V3", "Qwen-2.5-72B", "Llama-3.3-70B", "GPT-3.5 (baseline)"]
BASELINE_IDX = 3  # GPT-3.5 is appended last from the CSV

METRICS = ["TNCSI", "TNCSI_SP"]

# ============================================================
# HELPERS
# ============================================================

def load_data(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_paired(mapping: dict, metric: str, idx_a: int, idx_b: int):
    """Return two aligned arrays for papers where both models have valid scores."""
    a_vals, b_vals = [], []
    for data in mapping.values():
        vals = data[metric]
        if len(vals) > max(idx_a, idx_b):
            va, vb = vals[idx_a], vals[idx_b]
            if va != -1 and vb != -1:
                a_vals.append(va)
                b_vals.append(vb)
    return np.array(a_vals), np.array(b_vals)


def extract_all_valid(mapping: dict, metric: str, n_models: int):
    """Return (paper_ids, matrix) for papers with complete data across all models."""
    ids, rows = [], []
    for pid, data in mapping.items():
        vals = data[metric]
        if len(vals) == n_models and all(v != -1 for v in vals):
            ids.append(pid)
            rows.append(vals)
    return ids, np.array(rows) if rows else np.empty((0, n_models))


def spearman(a, b):
    if len(a) < 3:
        return float("nan"), float("nan")
    rho, pval = stats.spearmanr(a, b)
    return rho, pval


def cohens_d(a, b):
    """Cohen's d for paired differences (mean diff / pooled std)."""
    diff = a - b
    if np.std(diff) == 0:
        return 0.0
    return np.mean(diff) / np.std(diff)


def wilcoxon_test(a, b):
    """Wilcoxon signed-rank test for paired samples."""
    diff = a - b
    if np.all(diff == 0) or len(diff) < 5:
        return float("nan"), float("nan")
    try:
        stat, pval = stats.wilcoxon(a, b)
        return stat, pval
    except Exception:
        return float("nan"), float("nan")


def calculate_icc21(data_matrix: np.ndarray) -> float:
    """ICC(2,1) – two-way random effects, single measures (consistency)."""
    n, k = data_matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand_mean = np.mean(data_matrix)
    row_means = np.mean(data_matrix, axis=1)
    col_means = np.mean(data_matrix, axis=0)
    ms_rows = k * np.sum((row_means - grand_mean) ** 2) / (n - 1)
    ms_cols = n * np.sum((col_means - grand_mean) ** 2) / (k - 1)
    ss_res = np.sum(
        (data_matrix - row_means[:, None] - col_means[None, :] + grand_mean) ** 2
    )
    ms_res = ss_res / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_res + k * (ms_cols - ms_res) / n
    if denom == 0:
        return float("nan")
    return (ms_rows - ms_res) / denom


def cv_per_paper(matrix: np.ndarray) -> np.ndarray:
    """Coefficient of variation per paper (row)."""
    means = np.abs(np.mean(matrix, axis=1))
    stds = np.std(matrix, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(means > 0, stds / means, np.nan)
    return cv


def high_variance_papers(mapping, metric, n_models, top_n=20):
    """Return top_n paper IDs ranked by score range across models."""
    rows = []
    for pid, data in mapping.items():
        vals = data[metric]
        if len(vals) == n_models and all(v != -1 for v in vals):
            arr = np.array(vals)
            rows.append((pid, arr, arr.max() - arr.min()))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows[:top_n]


# ============================================================
# SECTION PRINTERS
# ============================================================

def sep(title="", width=72):
    if title:
        print(f"\n{'='*width}\n{title}\n{'='*width}")
    else:
        print("=" * width)


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_rank_correlation(mapping, dataset_name, metric):
    """
    Spearman ρ between each challenger model and the GPT-3.5 baseline.
    Also reports 95% CI via Fisher z-transform.
    """
    sep(f"[{dataset_name}] {metric} – Rank Correlation vs. GPT-3.5 Baseline")

    rows = []
    for idx, model in enumerate(MODEL_NAMES):
        if idx == BASELINE_IDX:
            continue
        a, b = extract_paired(mapping, metric, BASELINE_IDX, idx)
        if len(a) < 3:
            print(f"  {model}: insufficient paired data (n={len(a)})")
            continue
        rho, pval = spearman(a, b)

        # Fisher z 95% CI
        n = len(a)
        z = np.arctanh(rho)
        se = 1 / np.sqrt(n - 3) if n > 3 else float("nan")
        ci_lo = np.tanh(z - 1.96 * se)
        ci_hi = np.tanh(z + 1.96 * se)

        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        rows.append(
            {
                "Model": model,
                "n": n,
                "Spearman ρ": rho,
                "95% CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                "p-value": pval,
                "Sig.": sig,
            }
        )
        print(
            f"  {model:22s}  n={n:5d}  ρ={rho:.4f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
            f"p={pval:.4e} {sig}"
        )

    return rows


def analyze_distributional_shift(mapping, dataset_name, metric):
    """
    For each challenger model vs. GPT-3.5 baseline:
      - Mean score difference (challenger − baseline)
      - Wilcoxon signed-rank test (non-parametric)
      - Cohen's d effect size
      - Fraction of papers where challenger is strictly higher/lower
    Identifies whether LLMs produce systematically stricter or more lenient metrics.
    """
    sep(f"[{dataset_name}] {metric} – Distributional Shift vs. GPT-3.5 Baseline")

    rows = []
    for idx, model in enumerate(MODEL_NAMES):
        if idx == BASELINE_IDX:
            continue
        a_base, b_chal = extract_paired(mapping, metric, BASELINE_IDX, idx)
        if len(a_base) < 5:
            print(f"  {model}: insufficient data (n={len(a_base)})")
            continue

        diff = b_chal - a_base
        mean_diff = np.mean(diff)
        median_diff = np.median(diff)
        w_stat, w_pval = wilcoxon_test(a_base, b_chal)
        d = cohens_d(b_chal, a_base)

        pct_higher = np.mean(b_chal > a_base) * 100
        pct_lower = np.mean(b_chal < a_base) * 100
        pct_equal = 100 - pct_higher - pct_lower

        direction = "more lenient (↑)" if mean_diff > 0 else "stricter (↓)"
        sig = "***" if w_pval < 0.001 else ("**" if w_pval < 0.01 else ("*" if w_pval < 0.05 else "n.s."))

        rows.append(
            {
                "Model": model,
                "n": len(a_base),
                "Mean Δ": mean_diff,
                "Median Δ": median_diff,
                "Cohen's d": d,
                "Wilcoxon p": w_pval,
                "Sig.": sig,
                "Direction": direction,
                "% Higher": pct_higher,
                "% Lower": pct_lower,
            }
        )
        print(
            f"\n  {model}")
        print(
            f"    Mean Δ={mean_diff:+.4f}  Median Δ={median_diff:+.4f}  d={d:+.4f}"
        )
        print(
            f"    Wilcoxon p={w_pval:.4e} {sig}  →  {direction}"
        )
        print(
            f"    Papers: {pct_higher:.1f}% higher, {pct_lower:.1f}% lower, "
            f"{pct_equal:.1f}% equal"
        )

    return rows


def analyze_overall_consistency(mapping, dataset_name, metric):
    """
    Across all models simultaneously:
      - ICC(2,1)
      - Mean CV per paper
      - Mean range per paper
    """
    sep(f"[{dataset_name}] {metric} – Overall Cross-Model Consistency (All Models)")

    n_models = len(MODEL_NAMES)
    ids, matrix = extract_all_valid(mapping, metric, n_models)
    n_complete = len(ids)

    if n_complete < 10:
        print(f"  Only {n_complete} papers with complete data — skipping.")
        return {}

    icc = calculate_icc21(matrix)
    cv = cv_per_paper(matrix)
    ranges = matrix.max(axis=1) - matrix.min(axis=1)

    icc_label = (
        "Excellent" if icc > 0.9 else
        "Good" if icc > 0.75 else
        "Moderate" if icc > 0.5 else
        "Poor"
    )

    print(f"  Papers with complete data: {n_complete}")
    print(f"  ICC(2,1)   = {icc:.4f}  [{icc_label}]")
    print(f"  Mean CV    = {np.nanmean(cv):.4f}  (median {np.nanmedian(cv):.4f})")
    print(f"  Mean Range = {np.mean(ranges):.4f}  (median {np.median(ranges):.4f})")
    print(f"  Pct papers with Range > 0.10 : {np.mean(ranges > 0.10)*100:.1f}%")
    print(f"  Pct papers with Range > 0.20 : {np.mean(ranges > 0.20)*100:.1f}%")

    return {
        "n_complete": n_complete,
        "ICC": icc,
        "ICC_label": icc_label,
        "mean_CV": np.nanmean(cv),
        "mean_range": np.mean(ranges),
    }


def analyze_pairwise_correlations(mapping, dataset_name, metric):
    """
    Full Spearman correlation matrix across all model pairs.
    """
    sep(f"[{dataset_name}] {metric} – Pairwise Spearman Correlation Matrix")

    n = len(MODEL_NAMES)
    matrix = np.full((n, n), np.nan)

    for i, j in itertools.combinations(range(n), 2):
        a, b = extract_paired(mapping, metric, i, j)
        if len(a) >= 3:
            rho, _ = spearman(a, b)
            matrix[i, j] = matrix[j, i] = rho
    np.fill_diagonal(matrix, 1.0)

    # Print table
    col_w = 24
    print(f"\n  {'':24s}", end="")
    for name in MODEL_NAMES:
        print(f"{name[:12]:>14s}", end="")
    print()
    for i, name in enumerate(MODEL_NAMES):
        print(f"  {name[:24]:24s}", end="")
        for j in range(n):
            val = matrix[i, j]
            print(f"{'—':>14s}" if np.isnan(val) else f"{val:>14.4f}", end="")
        print()

    return matrix


def analyze_retrieval_divergence(mapping, dataset_name, metric):
    """
    Identify papers with the highest score variance across models.
    These are candidates for detailed qualitative analysis (Retrieval Divergence).
    Reports:
      - Top-20 high-variance papers
      - Distribution of per-paper range
    """
    sep(f"[{dataset_name}] {metric} – Retrieval Divergence (High-Variance Papers)")

    n_models = len(MODEL_NAMES)
    top_papers = high_variance_papers(mapping, metric, n_models, top_n=20)

    if not top_papers:
        print("  No papers with complete data.")
        return

    ranges = [r for _, _, r in top_papers]

    print(f"\n  Top-20 papers by score range across {n_models} models:")
    print(f"  {'Paper ID':>20s}  {'Range':>8s}  {'Min':>8s}  {'Max':>8s}  {'Scores (per model)'}")
    print(f"  {'-'*80}")
    for pid, arr, rng in top_papers:
        scores_str = "  ".join(f"{v:.3f}" for v in arr)
        print(f"  {pid:>20s}  {rng:8.4f}  {arr.min():8.4f}  {arr.max():8.4f}  [{scores_str}]")

    # Percentile summary of per-paper ranges (over all complete papers)
    _, all_matrix = extract_all_valid(mapping, metric, n_models)
    if all_matrix.shape[0] > 0:
        all_ranges = all_matrix.max(axis=1) - all_matrix.min(axis=1)
        for pct in [50, 75, 90, 95, 99]:
            print(f"  P{pct} range: {np.percentile(all_ranges, pct):.4f}")


# ============================================================
# MAIN
# ============================================================

def main():
    sep("RQ3 – STEP 2: METRIC STABILITY ANALYSIS ACROSS LLM KEYWORD GENERATORS")
    print(f"Models: {MODEL_NAMES}")
    print(f"Baseline: {MODEL_NAMES[BASELINE_IDX]}")
    print(f"Metrics: {METRICS}")
    print(f"Datasets: {list(DATASETS.keys())}")

    summary_rows = []

    for dataset_name, path in DATASETS.items():
        sep(f"DATASET: {dataset_name}", width=72)
        mapping = load_data(path)
        print(f"  Total papers: {len(mapping)}")

        for metric in METRICS:
            # --- (a) Rank Correlation ---
            analyze_rank_correlation(mapping, dataset_name, metric)

            # --- (b) Distributional Shift ---
            analyze_distributional_shift(mapping, dataset_name, metric)

            # --- (c) Overall Consistency (ICC + CV) ---
            consistency = analyze_overall_consistency(mapping, dataset_name, metric)

            # --- (d) Pairwise Correlation Matrix ---
            analyze_pairwise_correlations(mapping, dataset_name, metric)

            # --- (e) Retrieval Divergence ---
            analyze_retrieval_divergence(mapping, dataset_name, metric)

            if consistency:
                summary_rows.append(
                    {
                        "Dataset": dataset_name,
                        "Metric": metric,
                        "n (complete)": consistency.get("n_complete", "—"),
                        "ICC(2,1)": f"{consistency['ICC']:.4f}",
                        "ICC label": consistency.get("ICC_label", "—"),
                        "Mean CV": f"{consistency['mean_CV']:.4f}",
                        "Mean Range": f"{consistency['mean_range']:.4f}",
                    }
                )

    # ---- MASTER SUMMARY TABLE ----
    sep("MASTER SUMMARY TABLE")
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print(df.to_string(index=False))

    sep("INTERPRETATION GUIDE")
    print("""
Rank Correlation (Spearman ρ):
  ρ > 0.90  → Near-identical ranking; metric is robust to LLM choice.
  ρ 0.75–0.90 → Good agreement; minor reordering of borderline papers.
  ρ < 0.75  → Substantial reordering; metric is sensitive to LLM choice.

Distributional Shift (Wilcoxon + Cohen's d):
  Mean Δ > 0 → Challenger LLM generates broader keywords → larger cohorts
               → lower normalization baseline → TNCSI inflated ('more lenient').
  Mean Δ < 0 → Narrower keywords → smaller, more selective cohorts
               → higher baseline → TNCSI deflated ('stricter').
  |d| < 0.2  → Negligible practical effect.
  |d| 0.2–0.5 → Small effect.
  |d| > 0.5  → Moderate/large; the LLM choice materially changes absolute scores.

ICC(2,1):
  > 0.90 → Excellent agreement across all models.
  0.75–0.90 → Good; models are interchangeable for most purposes.
  0.50–0.75 → Moderate; LLM choice introduces non-trivial variance.
  < 0.50 → Poor; metric cannot be treated as model-agnostic.

Retrieval Divergence:
  Papers with large score ranges are candidates for qualitative analysis.
  Inspect their generated keywords to determine whether divergence stems from
  ambiguous titles, highly specialized topics, or cross-disciplinary scope.
""")


if __name__ == "__main__":
    main()