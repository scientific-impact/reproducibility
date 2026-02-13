#!/usr/bin/env python3
"""
Reproduce de Winter (2024) "Can ChatGPT be used to predict citation counts?"
==============================================================================
Applies the paper's methodology to LLM-scored abstracts from NAID v1/v2:
  1. Extract & clean 60-item scores from multiple LLM judges
  2. Descriptive statistics (Table 2 style)
  3. PCA with Varimax rotation → 3 components
  4. Spearman correlations between component scores and impact metrics (Table 3 style)
  5. Readability analysis comparison (Table 4 style)
  6. Antonym consistency check
  7. Cross-model comparison
"""

import pandas as pd
import numpy as np
import json
import re
import os
import sys
import warnings
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Ensure dataset dir is on path for item_mapping (run from any cwd)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from item_mapping import (
    COMPONENT_NAMES,
    get_marker_indices,
    get_antonym_pairs,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1.  CONFIGURATION
# ============================================================
# ITEMS must match the order that was in the LLM prompt when scores were collected.
# Current data (ver1, ver2, ver3, ver4) was collected with this order → use it for all.

ITEMS = [
    "1. Insightful", "2. Haphazard", "3. Well-sourced", "4. Disengaging", "5. Technical",
    "6. Balanced", "7. Superficial", "8. Ethical", "9. Concise", "10. Unreliable",
    "11. Innovative", "12. Subjective", "13. Circumlocutory", "14. Rigorous", "15. Dull",
    "16. Replicable", "17. Provocative", "18. Inaccessible", "19. Empirical", "20. Coherent",
    "21. Poorly-sourced", "22. Impactful", "23. Nontechnical", "24. Unstructured", "25. Engaging",
    "26. Narrow", "27. Methodical", "28. Unconvincing", "29. Exciting", "30. Speculation-driven",
    "31. Relevant", "32. Lax", "33. Well written", "34. Hypothesis-driven", "35. Inconsequential",
    "36. Accessible", "37. Authoritative", "38. Not well written", "39. Interdisciplinary",
    "40. Easy to understand", "41. Original", "42. Unbalanced", "43. Comprehensive",
    "44. Verbose", "45. Conventional", "46. To the point", "47. Unprovocative",
    "48. Structured", "49. Difficult to understand", "50. Irrelevant",
    "51. Controversial", "52. Non-replicable", "53. Objective", "54. Derivative",
    "55. Unethical", "56. Persuasive", "57. Incoherent", "58. Uninsightful",
    "59. Theoretical", "60. Exciting"
]

# Derived from current ITEMS order (works for ver1, ver2, ver3, ...)
MARKER_ITEMS = get_marker_indices(ITEMS)
ANTONYM_PAIRS = get_antonym_pairs(ITEMS)

BASE_DIR   = "/mnt/data/son/Reviewerly"
DATASET_DIR = f"{BASE_DIR}/dataset/ver3"
NAIDV1_DIR  = f"{BASE_DIR}/NAIP/v1_resource/NAIDv1"
NAIDV2_DIR  = f"{BASE_DIR}/NAIP/v2_resource/NAIDv2"
OUTPUT_DIR  = "/mnt/data/son/Reviewerly/analysis_results/ver3"

MODELS  = [ "GPT", "Deepseek", "Llama-3.1", "Llama-3.3"]
DATASETS = ["naidv1", "naidv2"]
SPLITS   = ["train", "test"]


# ============================================================
# 2.  SCORE EXTRACTION  (handles interleaved line-numbers bug)
# ============================================================

def _parse_response_linewise(response: str):
    """
    Parse LLM response line-by-line to extract (item_number, score) pairs.
    Handles formats: "1. 75", "1) 75", "1.75.0" (edge), "1: 75", etc.
    """
    pairs = []
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        # "N. score"  or  "N) score"  or  "N: score"
        m = re.match(r"^(\d{1,2})\s*[\.\)\:\-]\s*(\d+(?:\.\d+)?)\s*$", line)
        if m:
            item_no = int(m.group(1))
            score   = float(m.group(2))
            if 1 <= item_no <= 60 and 0 <= score <= 100:
                pairs.append((item_no, score))
    return pairs


def extract_clean_scores(raw_scores, llm_response):
    """
    Return a list of exactly 60 float scores (items 1-60), or None on failure.

    Strategy:
      1. Re-parse llm_response line-by-line (most reliable).
      2. Fall back: detect & de-interleave raw_scores.
      3. Fall back: take raw_scores[:60] if they look sane.
    """
    # --- Strategy 1: re-parse response ---
    if isinstance(llm_response, str) and llm_response.strip():
        pairs = _parse_response_linewise(llm_response)
        # if p[0] in pairs not in [1-60 order]
        if len(pairs) >= 55:
            # build a dict, fill in order 1..60
            d = {p[0]: p[1] for p in pairs}
            if len(set(d.keys()).intersection(set(range(1, 61)))) < 50:
                return [s * 10 if isinstance(s, (int, float)) and 0 < s < 10 else s for s in raw_scores]
            scores = [d.get(i, np.nan) for i in range(1, 61)]
            if sum(np.isnan(scores)) <= 5:      # tolerate a few missing
                # impute missing with item mean (will be overwritten later if needed)
                arr = np.array(scores)
                if np.any(np.isnan(arr)):
                    arr[np.isnan(arr)] = np.nanmean(arr)
                return arr.tolist()

    return None


# ============================================================
# 3.  READABILITY METRICS  (simple implementations)
# ============================================================

def _count_syllables(word):
    """Rough syllable count using vowel-group heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # silent-e
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def compute_readability(text):
    """Compute readability indices from raw text."""
    if not isinstance(text, str) or len(text.strip()) < 20:
        return {}

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z]+", text)

    n_sent = max(len(sentences), 1)
    n_words = max(len(words), 1)
    n_chars = sum(len(w) for w in words)
    syllables = [_count_syllables(w) for w in words]
    n_syl = sum(syllables)
    n_polysyl = sum(1 for s in syllables if s >= 3)

    wps = n_words / n_sent          # words per sentence
    spw = n_syl / n_words           # syllables per word
    cpw = n_chars / n_words         # chars per word

    flesch_re = 206.835 - 1.015 * wps - 84.6 * spw
    flesch_kg = 0.39 * wps + 11.8 * spw - 15.59
    ari       = 4.71 * cpw + 0.5 * wps - 21.43
    smog      = 1.0430 * np.sqrt(n_polysyl * (30.0 / n_sent)) + 3.1291 if n_sent > 0 else np.nan
    gunning   = 0.4 * (wps + 100.0 * n_polysyl / n_words)
    
    # Coleman-Liau
    L = (n_chars / n_words) * 100
    S = (n_sent / n_words) * 100
    coleman   = 0.0588 * L - 0.296 * S - 15.8

    return {
        "n_sentences": n_sent,
        "n_words": n_words,
        "n_characters": n_chars,
        "n_syllables": n_syl,
        "n_polysyllables": n_polysyl,
        "Flesch_Reading_Ease": flesch_re,
        "Flesch_Kincaid_Grade": flesch_kg,
        "ARI": ari,
        "SMOG": smog,
        "Gunning_Fog": gunning,
        "Coleman_Liau": coleman,
    }


# ============================================================
# 4.  DATA LOADING
# ============================================================

def load_llm_scores(dataset, split, model):
    """Load & clean LLM scores JSON → DataFrame with item_1 .. item_60 columns."""
    path = os.path.join(DATASET_DIR, split, dataset, model, "llm_scores.json")
    if not os.path.exists(path):
        print(f"  [SKIP] {path}")
        return None

    with open(path) as f:
        data = json.load(f)

    records, skipped = [], 0
    for entry in data:
        scores = extract_clean_scores(
            entry.get("raw_scores"),
            entry.get("llm_response", ""),
        )
        if scores is not None and len(scores) == 60:
            rec = {"id": entry["id"], "cites_json": entry.get("cites")}
            for i, s in enumerate(scores):
                rec[f"item_{i+1}"] = float(s)
            records.append(rec)
        else:
            skipped += 1

    print(f"  [{model:>10}] {len(records):>5} OK, {skipped:>4} skipped  ({path})")
    if not records:
        return None
    return pd.DataFrame(records)


def load_metadata(dataset, split):
    """Load original CSV with citation / review metadata."""
    if dataset == "naidv1":
        fname = "NAID_train_extrainfo.csv" if split == "train" else "NAID_test_extrainfo.csv"
        path = os.path.join(NAIDV1_DIR, fname)
    else:
        fname = "NAIDv2-train-with-cites.csv" if split == "train" else "NAIDv2-test-with-cites.csv"
        path = os.path.join(NAIDV2_DIR, fname)

    if not os.path.exists(path):
        print(f"  [SKIP] metadata not found: {path}")
        return None
    return pd.read_csv(path)


def load_llm_scores_combined(dataset, model):
    """Load train + test LLM scores and concatenate into one DataFrame."""
    dfs = []
    for split in SPLITS:
        df = load_llm_scores(dataset, split, model)
        if df is not None and len(df) > 0:
            dfs.append(df)
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  [{model:>10}] combined train+test: {len(combined)} rows  ({dataset})")
    return combined


def load_metadata_combined(dataset):
    """Load train + test metadata and concatenate into one DataFrame."""
    dfs = []
    for split in SPLITS:
        df = load_metadata(dataset, split)
        if df is not None and len(df) > 0:
            dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


# ============================================================
# 5.  PCA  +  VARIMAX ROTATION
# ============================================================

def varimax(Phi, max_iter=200, tol=1e-8):
    """
    Varimax rotation of a p × k loadings matrix Phi.
    Returns (rotated_loadings, rotation_matrix).
    """
    p, k = Phi.shape
    R = np.eye(k)
    for _ in range(max_iter):
        Lambda = Phi @ R
        # Compute the gradient
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lambda ** 3 - Lambda @ np.diag(np.sum(Lambda ** 2, axis=0)) / p)
        )
        R_new = u @ vt
        if np.max(np.abs(R_new - R)) < tol:
            R = R_new
            break
        R = R_new
    return Phi @ R, R


def run_pca_varimax(score_matrix, n_components=3):
    """
    Standardise → PCA → Varimax.
    Returns:
        component_scores  (n × k)
        rotated_loadings   (60 × k)
        eigenvalues        (all 60)
        var_explained      (first k, pre-rotation)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(score_matrix)

    # Full PCA for scree plot
    pca_full = PCA()
    pca_full.fit(X)
    eigenvalues = pca_full.explained_variance_

    # Reduced PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)
    raw_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Varimax rotation
    rot_loadings, rot_matrix = varimax(raw_loadings)

    # Component scores: X @ pinv(L^T)
    comp_scores = X @ np.linalg.pinv(rot_loadings.T)

    # Auto-label and reorder components for consistency
    rot_loadings, comp_scores, label_order = auto_label_components(rot_loadings, comp_scores)

    return comp_scores, rot_loadings, eigenvalues, pca.explained_variance_ratio_


def auto_label_components(rot_loadings, comp_scores):
    """
    Identify which PCA component corresponds to each theoretical construct
    by checking marker item loadings. Reorder to match COMPONENT_NAMES order.
    
    Handles sign-flipping: PCA can return components multiplied by -1.
    We check both orientations and use the one with better marker match.
    
    Returns:
        reordered_loadings (60 × k)
        reordered_scores (n × k)
        label_order: list of assigned labels in original component order
    """
    n_components = rot_loadings.shape[1]
    
    # First, determine optimal sign for each component
    # by checking if flipping improves marker alignment
    sign_multipliers = np.ones(n_components)
    
    for j in range(n_components):
        loadings = rot_loadings[:, j]
        
        # Compute total marker alignment score for original and flipped
        original_score = 0
        flipped_score = 0
        
        for comp_name in COMPONENT_NAMES:
            markers = MARKER_ITEMS[comp_name]
            # Original: positive markers should be positive, negative should be negative
            pos_orig = np.mean(loadings[markers["positive"]])
            neg_orig = -np.mean(loadings[markers["negative"]])
            original_score += pos_orig + neg_orig
            
            # Flipped: signs reversed
            pos_flip = np.mean(-loadings[markers["positive"]])
            neg_flip = -np.mean(-loadings[markers["negative"]])
            flipped_score += pos_flip + neg_flip
        
        # Use the sign that gives better overall alignment
        if flipped_score > original_score:
            sign_multipliers[j] = -1
    
    # Apply sign corrections
    rot_loadings_corrected = rot_loadings * sign_multipliers
    comp_scores_corrected = comp_scores * sign_multipliers
    
    # Now score each corrected component against each marker set
    scores_matrix = np.zeros((n_components, len(COMPONENT_NAMES)))
    
    for j in range(n_components):
        loadings = rot_loadings_corrected[:, j]
        for i, comp_name in enumerate(COMPONENT_NAMES):
            markers = MARKER_ITEMS[comp_name]
            # Positive markers should have positive loadings
            pos_score = np.mean(loadings[markers["positive"]])
            # Negative markers should have negative loadings (so negate them)
            neg_score = -np.mean(loadings[markers["negative"]])
            scores_matrix[j, i] = pos_score + neg_score
    
    # Use greedy assignment to find best matching
    assignment = []
    used_components = set()
    
    for target_idx in range(len(COMPONENT_NAMES)):
        best_score = -np.inf
        best_comp = None
        for comp_idx in range(n_components):
            if comp_idx in used_components:
                continue
            if scores_matrix[comp_idx, target_idx] > best_score:
                best_score = scores_matrix[comp_idx, target_idx]
                best_comp = comp_idx
        if best_comp is not None:
            assignment.append(best_comp)
            used_components.add(best_comp)
    
    # Reorder loadings and scores
    reordered_loadings = rot_loadings_corrected[:, assignment]
    reordered_scores = comp_scores_corrected[:, assignment]
    
    # Create label order for reporting
    label_order = [COMPONENT_NAMES[i] for i in range(len(assignment))]
    
    return reordered_loadings, reordered_scores, label_order


# ============================================================
# 6.  CORRELATION HELPERS
# ============================================================

def spearman_table(comp_scores, df, target_cols, comp_names=COMPONENT_NAMES):
    """
    Compute Spearman ρ between each component and each target variable.
    Returns a DataFrame with columns: Component, Target, rho, p, n.
    """
    rows = []
    for j, cname in enumerate(comp_names[: comp_scores.shape[1]]):
        c = comp_scores[:, j]
        for col in target_cols:
            if col not in df.columns:
                continue
            y = pd.to_numeric(df[col], errors="coerce").values.astype(float)
            mask = np.isfinite(c) & np.isfinite(y)
            n = int(mask.sum())
            if n < 10:
                continue
            rho, p = stats.spearmanr(c[mask], y[mask])
            rows.append({"Component": cname, "Target": col, "rho": rho, "p": p, "n": n})
    return pd.DataFrame(rows)


def antonym_consistency(rot_loadings):
    """
    Check that loadings of items 1-30 correlate negatively with items 31-60.
    Returns per-component Pearson r between positive and antonym loadings.
    """
    k = rot_loadings.shape[1]
    results = {}
    for j in range(k):
        pos = rot_loadings[:30, j]
        neg = rot_loadings[30:, j]
        r, _ = stats.pearsonr(pos, neg)
        results[COMPONENT_NAMES[j]] = r
    return results


# ============================================================
# 7.  VISUALISATION
# ============================================================

def plot_scree(eigenvalues, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, len(eigenvalues) + 1)
    total = eigenvalues.sum()
    pct = eigenvalues / total * 100
    ax.plot(x, eigenvalues, "o-", markersize=4, linewidth=1.2)
    for i in range(min(6, len(eigenvalues))):
        ax.annotate(f"{pct[i]:.1f}%", (x[i], eigenvalues[i]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8, color="blue")
    ax.set_xlabel("Component number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_loadings_heatmap(loadings_df, title, save_path):
    """Heatmap of Varimax-rotated loadings (items × components)."""
    fig, ax = plt.subplots(figsize=(8, 18))
    data = loadings_df[COMPONENT_NAMES].values
    labels = [f"{i+1}. {ITEMS[i]}" for i in range(60)]
    sns.heatmap(data, yticklabels=labels, xticklabels=COMPONENT_NAMES,
                center=0, cmap="RdBu_r", vmin=-1, vmax=1,
                linewidths=0.2, ax=ax, fmt=".2f",
                annot=False)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_correlation_comparison(all_results, target_col, save_path):
    """Bar chart comparing Spearman ρ across models for one target variable."""
    rows = []
    for key, res in all_results.items():
        corr = res["correlations"]
        if corr is None or len(corr) == 0:
            continue
        sub = corr[corr["Target"] == target_col]
        for _, r in sub.iterrows():
            rows.append({
                "Model": res["model"],
                "Dataset": res["dataset"],
                "Split": res["split"],
                "Component": r["Component"],
                "rho": r["rho"],
            })
    if not rows:
        return
    df_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 5))
    # Group by model+split
    df_plot["group"] = df_plot["Model"] + " (" + df_plot["Split"] + ")"
    sns.barplot(data=df_plot, x="group", y="rho", hue="Component", ax=ax)
    ax.set_title(f"Spearman ρ  with  '{target_col}'  across models")
    ax.set_ylabel("Spearman ρ")
    ax.set_xlabel("")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8, loc="upper right")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# 8.  MAIN ANALYSIS FOR ONE CONFIGURATION
# ============================================================

def analyze_one(dataset, split, model, out_dir, df_scores=None, df_meta=None):
    """Full pipeline for one (dataset, split, model) triple.
    If df_scores and df_meta are provided, they are used instead of loading by split.
    """
    tag = f"{dataset}_{split}_{model}"
    print(f"\n{'='*70}")
    print(f"  {tag}")
    print(f"{'='*70}")

    # --- Load (unless provided) ---
    if df_scores is None:
        df_scores = load_llm_scores(dataset, split, model)
    if df_scores is None or len(df_scores) < 30:
        print("  Not enough data – skipping.")
        return None

    if df_meta is None:
        df_meta = load_metadata(dataset, split)

    # Merge
    if df_meta is not None:
        df = df_scores.merge(df_meta, on="id", how="left", suffixes=("", "_meta"))
    else:
        df = df_scores.copy()

    # Use cites from JSON if CSV cites is missing
    if "cites" not in df.columns and "cites_json" in df.columns:
        df["cites"] = df["cites_json"]

    # --- Filter out invalid rows (cites=-1, TNCSI=-1, TNCSI_SP=-1) ---
    n_before = len(df)
    for col in ["cites", "TNCSI", "TNCSI_SP"]:
        if col in df.columns:
            df = df[df[col] != -1]
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} rows with -1 values in cites/TNCSI/TNCSI_SP")
    df = df.reset_index(drop=True)
    print(f"  Remaining samples after filtering: {len(df)}")
    print(df.head(5))

    # --- Score matrix ---
    item_cols = [f"item_{i}" for i in range(1, 61)]
    score_mat = df[item_cols].values.astype(float)
    valid = ~np.any(np.isnan(score_mat), axis=1)
    score_mat = score_mat[valid]
    df = df.loc[valid].reset_index(drop=True)
    n = len(score_mat)
    print(f"  Valid samples: {n}")

    if n < 60:
        print("  Too few samples for reliable PCA – skipping.")
        return None

    # --- Descriptive statistics (Table 2 style) ---
    desc = pd.DataFrame({
        "No": range(1, 61), "Item": ITEMS,
        "Mean": score_mat.mean(axis=0).round(2),
        "SD": score_mat.std(axis=0).round(2),
    })

    # --- PCA + Varimax ---
    comp_scores, rot_load, eigenvalues, var_exp = run_pca_varimax(score_mat, n_components=3)

    for j, cn in enumerate(COMPONENT_NAMES):
        desc[cn] = rot_load[:, j].round(3)
        df[cn] = comp_scores[:, j]

    # --- Antonym consistency ---
    ant_cons = antonym_consistency(rot_load)
    print(f"  Antonym loadings correlation: { {k: round(v,3) for k,v in ant_cons.items()} }")

    # --- Target variables ---
    if dataset == "naidv1":
        targets = ["cites", "TNCSI", "TNCSI_SP", "Ref_num", "RQM", "SMP", "ARQ",
                    "OA", "is_practical", "new_task", "new_dataset", "SOTA", "is_broad"]
    else:
        targets = ["score_mean", "score_weighted", "score_median", "accept", "cites", "TNCSI", "TNCSI_SP"]

    # always include cites if present
    if "cites" in df.columns and "cites" not in targets:
        targets = ["cites"] + targets

    available = [c for c in targets if c in df.columns]

    # --- Spearman correlations (Table 3 style) ---
    corr_df = spearman_table(comp_scores, df, available)

    if len(corr_df):
        pivot = corr_df.pivot(index="Target", columns="Component", values="rho")
        pivot_p = corr_df.pivot(index="Target", columns="Component", values="p")
        print(f"\n  Spearman ρ  (n ≈ {n}):\n")
        print(pivot.round(3).to_string())

    # --- Readability (Table 4 style) ---
    abstract_col = "abstract" if "abstract" in df.columns else None
    read_df = None
    if abstract_col:
        read_rows = df[abstract_col].apply(compute_readability)
        read_df = pd.DataFrame(read_rows.tolist())
        for c in read_df.columns:
            df[c] = read_df[c]
        read_targets = [
            "n_sentences", "n_words", "n_characters", "n_syllables", "n_polysyllables",
            "ARI", "SMOG", "Flesch_Kincaid_Grade", "Flesch_Reading_Ease",
            "Coleman_Liau", "Gunning_Fog",
        ]
        read_avail = [c for c in read_targets if c in df.columns]
        # Correlations: readability → impact targets
        read_impact_corr = []
        for rc in read_avail:
            for tc in available:
                y1 = pd.to_numeric(df[rc], errors="coerce").values.astype(float)
                y2 = pd.to_numeric(df[tc], errors="coerce").values.astype(float)
                mask = np.isfinite(y1) & np.isfinite(y2)
                if mask.sum() < 10:
                    continue
                rho, p = stats.spearmanr(y1[mask], y2[mask])
                read_impact_corr.append({"Readability": rc, "Target": tc, "rho": rho, "p": p})
        read_impact_df = pd.DataFrame(read_impact_corr)

        # Correlations: readability → component scores (Table 5 style)
        read_comp_corr = []
        for rc in read_avail:
            for j, cn in enumerate(COMPONENT_NAMES):
                y1 = pd.to_numeric(df[rc], errors="coerce").values.astype(float)
                c = comp_scores[:, j]
                mask = np.isfinite(y1) & np.isfinite(c)
                if mask.sum() < 10:
                    continue
                rho, p = stats.spearmanr(y1[mask], c[mask])
                read_comp_corr.append({"Readability": rc, "Component": cn, "rho": rho, "p": p})
        read_comp_df = pd.DataFrame(read_comp_corr)
    else:
        read_impact_df = pd.DataFrame()
        read_comp_df = pd.DataFrame()

    # --- Save artefacts ---
    tag_dir = os.path.join(out_dir, tag)
    os.makedirs(tag_dir, exist_ok=True)

    desc.to_csv(os.path.join(tag_dir, "table2_loadings.csv"), index=False)
    if len(corr_df):
        corr_df.to_csv(os.path.join(tag_dir, "table3_correlations.csv"), index=False)
    if len(read_impact_df):
        read_impact_df.to_csv(os.path.join(tag_dir, "table4_readability_impact.csv"), index=False)
    if len(read_comp_df):
        read_comp_df.to_csv(os.path.join(tag_dir, "table5_readability_components.csv"), index=False)

    plot_scree(eigenvalues, f"Scree Plot – {tag}", os.path.join(tag_dir, "scree_plot.png"))
    plot_loadings_heatmap(desc, f"Varimax Loadings – {tag}", os.path.join(tag_dir, "loadings_heatmap.png"))

    return {
        "dataset": dataset, "split": split, "model": model,
        "n": n,
        "desc": desc,
        "correlations": corr_df,
        "antonym_consistency": ant_cons,
        "var_explained": var_exp,
        "eigenvalues": eigenvalues,
        "read_impact": read_impact_df,
        "read_comp": read_comp_df,
    }


def analyze_one_combined(dataset, model, out_dir):
    """Full pipeline for one (dataset, model) with train+test combined."""
    df_scores = load_llm_scores_combined(dataset, model)
    if df_scores is None or len(df_scores) < 30:
        return None
    df_meta = load_metadata_combined(dataset)
    return analyze_one(dataset, "all", model, out_dir, df_scores=df_scores, df_meta=df_meta)


# ============================================================
# 9.  CROSS-MODEL COMPARISON
# ============================================================

def cross_model_summary(all_results, out_dir):
    """Build a summary table comparing Spearman ρ across all configurations."""
    rows = []
    for key, res in all_results.items():
        if res is None:
            continue
        corr = res["correlations"]
        if corr is None or len(corr) == 0:
            continue
        for _, r in corr.iterrows():
            rows.append({
                "Dataset": res["dataset"],
                "Split": res["split"],
                "Model": res["model"],
                "n": res["n"],
                "Component": r["Component"],
                "Target": r["Target"],
                "rho": r["rho"],
                "p": r["p"],
            })
    if not rows:
        print("No results to summarise.")
        return

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "cross_model_summary.csv"), index=False)
    print(f"\n  Saved cross-model summary → {os.path.join(out_dir, 'cross_model_summary.csv')}")

    # --- Pretty-print for key targets ---
    for dataset in DATASETS:
        sub = summary[summary["Dataset"] == dataset]
        if len(sub) == 0:
            continue
        key_targets = sub["Target"].unique()[:6]  # top targets
        for tgt in key_targets:
            sub_t = sub[sub["Target"] == tgt]
            if len(sub_t) == 0:
                continue
            piv = sub_t.pivot_table(index=["Model", "Split"], columns="Component", values="rho")
            print(f"\n  --- {dataset} | Target = {tgt} ---")
            print(piv.round(3).to_string())

    # --- Plots ---
    for dataset in DATASETS:
        sub = summary[summary["Dataset"] == dataset]
        if len(sub) == 0:
            continue
        key_targets = sub["Target"].unique()[:3]
        for tgt in key_targets:
            save = os.path.join(out_dir, f"compare_{dataset}_{tgt}.png")
            sub_t = sub[(sub["Target"] == tgt)]
            if len(sub_t) == 0:
                continue
            fig, ax = plt.subplots(figsize=(12, 5))
            sub_t = sub_t.copy()
            sub_t["group"] = sub_t["Model"] + " (" + sub_t["Split"] + ")"
            sns.barplot(data=sub_t, x="group", y="rho", hue="Component", ax=ax)
            ax.set_title(f"Spearman ρ  →  {tgt}   [{dataset}]")
            ax.set_ylabel("ρ")
            ax.axhline(0, color="k", lw=0.5)
            plt.xticks(rotation=25, ha="right", fontsize=9)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(save, dpi=150)
            plt.close(fig)
            print(f"  Saved: {save}")


# ============================================================
# 10.  ANTONYM CONSISTENCY COMPARISON
# ============================================================

def antonym_summary(all_results, out_dir):
    rows = []
    for key, res in all_results.items():
        if res is None:
            continue
        for comp, r in res["antonym_consistency"].items():
            rows.append({
                "Dataset": res["dataset"], "Split": res["split"],
                "Model": res["model"], "Component": comp, "antonym_r": round(r, 3),
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "antonym_consistency.csv"), index=False)
        print(f"\n  Antonym Consistency:")
        piv = df.pivot_table(index=["Model", "Dataset"], columns="Component", values="antonym_r")
        print(piv.round(3).to_string())


# ============================================================
# 11.  INDIVIDUAL-ITEM CORRELATIONS  (supplement)
# ============================================================

def item_level_correlations(dataset, split, model, out_dir, df_scores=None, df_meta=None):
    """Spearman ρ between each of the 60 items and citation count.
    If df_scores/df_meta are provided (e.g. combined train+test), uses those.
    """
    tag = f"{dataset}_{split}_{model}"
    if df_scores is None:
        df_scores = load_llm_scores(dataset, split, model)
    if df_scores is None or len(df_scores) < 30:
        return None

    if df_meta is None:
        df_meta = load_metadata(dataset, split)
    if df_meta is not None:
        df = df_scores.merge(df_meta, on="id", how="left", suffixes=("", "_meta"))
    else:
        df = df_scores.copy()

    if "cites" not in df.columns and "cites_json" in df.columns:
        df["cites"] = df["cites_json"]

    target = "cites" if "cites" in df.columns else ("score_mean" if "score_mean" in df.columns else None)
    if target is None:
        return None

    y = pd.to_numeric(df[target], errors="coerce").values.astype(float)
    rows = []
    for i in range(60):
        col = f"item_{i+1}"
        x = df[col].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            continue
        rho, p = stats.spearmanr(x[mask], y[mask])
        rows.append({"Item_No": i + 1, "Item": ITEMS[i], "rho": rho, "p": p})

    if rows:
        item_df = pd.DataFrame(rows).sort_values("rho", ascending=False)
        save_path = os.path.join(out_dir, tag, "item_level_corr.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        item_df.to_csv(save_path, index=False)
        print(f"\n  Top-5 items correlated with {target} ({tag}):")
        print(item_df.head(5).to_string(index=False))
        print(f"  Bottom-5:")
        print(item_df.tail(5).to_string(index=False))
    return item_df if rows else None


# ============================================================
# 12.  MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    # Run analysis with train+test combined per (dataset, model)
    for dataset in DATASETS:
        for model in MODELS:
            key = f"{dataset}_all_{model}"
            result = analyze_one_combined(dataset, model, OUTPUT_DIR)
            all_results[key] = result

    # Cross-model summary
    print("\n\n" + "=" * 80)
    print("  CROSS-MODEL SUMMARY")
    print("=" * 80)
    cross_model_summary(all_results, OUTPUT_DIR)
    antonym_summary(all_results, OUTPUT_DIR)

    # Item-level correlations (on combined train+test)
    print("\n\n" + "=" * 80)
    print("  ITEM-LEVEL CORRELATIONS (train+test combined)")
    print("=" * 80)
    for dataset in DATASETS:
        for model in MODELS:
            df_scores = load_llm_scores_combined(dataset, model)
            df_meta = load_metadata_combined(dataset)
            if df_scores is not None:
                item_level_correlations(dataset, "all", model, OUTPUT_DIR,
                                        df_scores=df_scores, df_meta=df_meta)

    print("\n\nDone.  All outputs saved to:", OUTPUT_DIR)
    return all_results


if __name__ == "__main__":
    results = main()