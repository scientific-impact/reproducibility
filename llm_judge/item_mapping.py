"""
Canonical item names and mappings for de Winter (2024) 60-item scale.
Use this so MARKER_ITEMS and ANTONYM_PAIRS work for any ITEMS order (ver1, ver2, ver3, ...).

CRITICAL - Prompt order vs analysis labels:
  The ITEMS list used in analysis MUST match the item order that was in the LLM prompt
  when the scores were collected. Position 1 in the JSON = first item in the prompt, etc.
  If you use a different ITEMS order in analysis than in the prompt, you get wrong labels
  (e.g. "Derivative" instead of "Coherent" at position 15) and spurious sign-flips in PCA.
  To study order sensitivity: re-run LLM scoring with a different prompt order, save to
  a separate path, then use the matching ITEMS list when analyzing that path.
"""

import re

# ============================================================
# Canonical prompt order used when current score files were collected.
# Original paper / ver1: positive traits 1-30, antonyms 31-60.
# All analysis scripts (ver1, ver2, ver3, ver4) must use this when
# analyzing data that was generated with this prompt order.
# ============================================================
ITEMS_PROMPT_ORDER_VER1 = [
    "Engaging", "Controversial", "Rigorous", "Innovative", "Accessible",
    "Methodical", "Concise", "Persuasive", "Comprehensive", "Insightful",
    "Relevant", "Objective", "Replicable", "Structured", "Coherent",
    "Original", "Balanced", "Authoritative", "Impactful", "Interdisciplinary",
    "Well-sourced", "Technical", "Provocative", "Hypothesis-driven", "Ethical",
    "Difficult to understand", "Exciting", "Not well written", "Theoretical", "To the point",
    "Disengaging", "Uncontroversial", "Lax", "Conventional", "Inaccessible",
    "Haphazard", "Verbose", "Unconvincing", "Superficial", "Uninsightful",
    "Irrelevant", "Subjective", "Non-replicable", "Unstructured", "Incoherent",
    "Derivative", "Unbalanced", "Unreliable", "Inconsequential", "Narrow",
    "Poorly-sourced", "Nontechnical", "Unprovocative", "Speculation-driven", "Unethical",
    "Easy to understand", "Dull", "Well written", "Empirical", "Circumlocutory",
]

# ============================================================
# Canonical 30 antonym pairs: (positive_trait, negative_trait)
# Positive = "desirable" pole (e.g. Engaging, Rigorous).
# ============================================================
CANONICAL_ANTONYM_PAIRS_NAMES = [
    ("Engaging", "Disengaging"),
    ("Controversial", "Uncontroversial"),
    ("Rigorous", "Lax"),
    ("Innovative", "Conventional"),
    ("Accessible", "Inaccessible"),
    ("Methodical", "Haphazard"),
    ("Concise", "Verbose"),
    ("Persuasive", "Unconvincing"),
    ("Comprehensive", "Superficial"),
    ("Insightful", "Uninsightful"),
    ("Relevant", "Irrelevant"),
    ("Objective", "Subjective"),
    ("Replicable", "Non-replicable"),
    ("Structured", "Unstructured"),
    ("Coherent", "Incoherent"),
    ("Original", "Derivative"),
    ("Balanced", "Unbalanced"),
    ("Authoritative", "Unreliable"),
    ("Impactful", "Inconsequential"),
    ("Interdisciplinary", "Narrow"),
    ("Well-sourced", "Poorly-sourced"),
    ("Technical", "Nontechnical"),
    ("Provocative", "Unprovocative"),
    ("Hypothesis-driven", "Speculation-driven"),
    ("Ethical", "Unethical"),
    ("Difficult to understand", "Easy to understand"),
    ("Exciting", "Dull"),
    ("Not well written", "Well written"),
    ("Theoretical", "Empirical"),
    ("To the point", "Circumlocutory"),
]

# Marker items for auto-labeling PCA components (by canonical name).
# Positive = items that should load positively on the component;
# Negative = items that should load negatively.
MARKER_ITEMS_BY_NAME = {
    "Quality & Reliability": {
        "positive": ["Rigorous", "Methodical", "Objective", "Replicable", "Structured", "Well-sourced"],
        "negative": ["Lax", "Haphazard", "Non-replicable", "Unstructured", "Unreliable"],
    },
    "Accessibility & Understandability": {
        "positive": ["Accessible", "Concise", "Coherent", "To the point", "Easy to understand"],
        "negative": ["Difficult to understand", "Inaccessible", "Verbose", "Incoherent", "Circumlocutory"],
    },
    "Novelty & Engagement": {
        "positive": ["Engaging", "Innovative", "Original", "Impactful", "Exciting"],
        "negative": ["Disengaging", "Conventional", "Derivative", "Inconsequential", "Dull"],
    },
}

COMPONENT_NAMES = [
    "Quality & Reliability",
    "Accessibility & Understandability",
    "Novelty & Engagement",
]


def _normalize_item_label(s):
    """Strip leading 'N. ' from item string to get canonical name."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"^\d+\.\s*", "", s).strip()


def build_name_to_index(items_list):
    """
    Build mapping from canonical item name -> 0-based index for the given ITEMS list.
    items_list: list of 60 strings, e.g. ["1. Circumlocutory", "2. Empirical", ...]
    """
    out = {}
    for i, raw in enumerate(items_list):
        name = _normalize_item_label(raw)
        if name:
            out[name] = i
    return out


def get_marker_indices(items_list, marker_by_name=None):
    """
    Return MARKER_ITEMS structure with 0-based indices for the given ITEMS list.
    marker_by_name: defaults to MARKER_ITEMS_BY_NAME.
    """
    name_to_idx = build_name_to_index(items_list)
    marker_by_name = marker_by_name or MARKER_ITEMS_BY_NAME
    result = {}
    for comp_name, d in marker_by_name.items():
        result[comp_name] = {
            "positive": [name_to_idx[n] for n in d["positive"] if n in name_to_idx],
            "negative": [name_to_idx[n] for n in d["negative"] if n in name_to_idx],
        }
    return result


def get_antonym_pairs(items_list, pair_names=None):
    """
    Return list of (index_a, index_b) for each canonical antonym pair,
    in order of CANONICAL_ANTONYM_PAIRS_NAMES. Either index can be the
    "positive" or "negative" trait depending on ITEMS order.
    pair_names: defaults to CANONICAL_ANTONYM_PAIRS_NAMES.
    """
    name_to_idx = build_name_to_index(items_list)
    pair_names = pair_names or CANONICAL_ANTONYM_PAIRS_NAMES
    pairs = []
    for pos_name, neg_name in pair_names:
        i_pos = name_to_idx.get(pos_name)
        i_neg = name_to_idx.get(neg_name)
        if i_pos is not None and i_neg is not None:
            pairs.append((i_pos, i_neg))
    return pairs
