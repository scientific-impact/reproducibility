# cd /mnt/data/son/Reviewerly/NAIP/v1_resource/TKPD
# export OPENROUTER_API_KEY="sk-or-..."
# python ned.py
"""
NED (Normalized Edit Distance) comparison across multiple LLMs via OpenRouter.
Reads TKPD.csv, runs keyword extraction with each model, reports mean NED per model.
Uses OpenAI client (OpenRouter-compatible) and batch concurrency via ThreadPoolExecutor.
"""
import os
import csv
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import Levenshtein
from retry import retry
from tqdm import tqdm
from openai import OpenAI

# OpenRouter: OpenAI-compatible API
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_BASE = os.environ.get("OPENROUTER_API_BASE", OPENROUTER_BASE_URL)
API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


# ---------------------------------------------------------------------------
# Models to compare (OpenRouter model IDs; many support vLLM backends)
# See https://openrouter.ai/models
# ---------------------------------------------------------------------------
OPENROUTER_MODELS = {
    "gpt-4o": "openai/gpt-4o:floor",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo:floor",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:floor",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:floor",
    "qwen-2.5-7b": "qwen/qwen-2.5-7b-instruct:floor",
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct:floor",
    "deepseek": "deepseek/deepseek-chat-v3-0324:floor",
    # vLLM-hosted variants (provider-dependent; use if you have vLLM endpoints on OpenRouter)
    # "qwen-2.5-7b-vllm": "openrouter/qwen-2.5-7b-instruct:free",  # example free tier
}


def normalized_edit_distance(str1: str, str2: str) -> float:
    str1 = str1.strip().lower()
    str2 = str2.strip().lower()
    edit_distance = Levenshtein.distance(str1, str2)
    max_length = max(len(str1), len(str2))
    return edit_distance / max_length if max_length != 0 else 0.0


DEFAULT_SYS = (
    "You are a profound researcher who is good at identifying the topic key phrase from paper's title and "
    "abstract. Ensure that the topic key phrase precisely defines the research area within the article. "
    "For effective academic searching, such as on Google Scholar, the field should be specifically targeted "
    "rather than broadly categorized. For instance, use 'image classification' instead of the general "
    "'computer vision' to enhance relevance and searchability of related literature."
)

DEFAULT_USR_PROMPT = (
    "Identify the research field from the given title and abstract. "
    "You MUST respond with the keyword ONLY in this format: xxx"
)

EXTRA_ABS = """
Given Title: Large Selective Kernel Network for Remote Sensing Object Detection
Given Abstract: Recent research on remote sensing object detection has largely focused on improving the representation of oriented bounding boxes but has overlooked the unique prior knowledge presented in remote sensing scenarios. Such prior knowledge can be useful because tiny remote sensing objects may be mistakenly detected without referencing a sufficiently long-range context, which can vary for different objects. This paper considers these priors and proposes the lightweight Large Selective Kernel Network (LSKNet). LSKNet can dynamically adjust its large spatial receptive field to better model the ranging context of various objects in remote sensing scenarios. To our knowledge, large and selective kernel mechanisms have not been previously explored in remote sensing object detection. Without bells and whistles, our lightweight LSKNet sets new state-of-the-art scores on standard benchmarks, i.e., HRSC2016 (98.46% mAP), DOTA-v1.0 (81.85% mAP), and FAIR1M-v1.0 (47.87% mAP)."""


def _build_messages(
    title: str,
    abstract: str | None,
    *,
    sys_content: str,
    usr_prompt: str,
    extra_prompt: bool,
) -> list[dict]:
    messages = [{"role": "system", "content": sys_content}]
    if extra_prompt:
        messages += [
            {"role": "user", "content": f"{usr_prompt}\n\n{EXTRA_ABS}"},
            {"role": "assistant", "content": "remote sensing object detection"},
        ]
    content = f"{usr_prompt}\nGiven Title: {title}\n"
    if abstract:
        content += f"Given Abstract: {abstract}"
    messages.append({"role": "user", "content": content})
    return messages


@retry(delay=2, tries=3)
def get_field(
    client: OpenAI,
    title: str,
    abstract: str | None,
    *,
    model_id: str,
    sys_content: str | None = None,
    usr_prompt: str | None = None,
    extra_prompt: bool = True,
    temperature: float = 0,
) -> str:
    """Call OpenRouter model to extract research field keyword from title/abstract."""
    sys_content = sys_content or DEFAULT_SYS
    usr_prompt = usr_prompt or DEFAULT_USR_PROMPT
    messages = _build_messages(title, abstract, sys_content=sys_content, usr_prompt=usr_prompt, extra_prompt=extra_prompt)

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def load_rows(csv_path: str, encoding: str = "utf-8") -> list[list[str]]:
    """Load CSV rows; fallback to gbk if utf-8 fails."""
    path = Path(csv_path)
    if not path.exists():
        path = Path(__file__).resolve().parent.parent.parent.parent / "dataset" / "TKPD.csv"
    for enc in (encoding, "gbk", "utf-8-sig"):
        try:
            with open(path, "r", newline="", encoding=enc) as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                return []
            # skip header if it looks like header
            if len(rows) > 1 and rows[0][0].lower() == "title":
                rows = rows[1:]
            return [r for r in rows if len(r) >= 3 and (r[0].strip() or r[1].strip())]
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    raise FileNotFoundError(f"Cannot read {csv_path}")


def _query_one(
    client: OpenAI,
    model_id: str,
    usr_prompt: str,
    idx: int,
    row: list[str],
) -> tuple[int, str]:
    """Single-row query for thread pool; returns (index, pred_kwd)."""
    title = row[0]
    abstract = (row[1] if len(row) > 1 else "") or None
    try:
        pred = get_field(client, title, abstract, model_id=model_id, usr_prompt=usr_prompt)
        return idx, pred
    except Exception:
        return idx, ""


def run_ned_for_model(
    rows: list[list[str]],
    model_key: str,
    model_id: str,
    usr_prompt: str | None = None,
    batch_size: int = 8,
    verbose: bool = True,
) -> tuple[float, list[float]]:
    """Run keyword extraction for one model on all rows in batches; return mean NED and list of NEDs."""
    usr_prompt = usr_prompt or DEFAULT_USR_PROMPT
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    pred_by_idx: dict[int, str] = {}

    for start in tqdm(range(0, len(rows), batch_size), desc=model_key, disable=not verbose):
        batch = [(start + i, rows[start + i]) for i in range(min(batch_size, len(rows) - start))]

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future2idx = {
                executor.submit(_query_one, client, model_id, usr_prompt, idx, row): idx
                for idx, row in batch
            }
            for future in as_completed(future2idx):
                try:
                    idx, pred_kwd = future.result()
                    pred_by_idx[idx] = pred_kwd
                except Exception as e:
                    if verbose:
                        tqdm.write(f"Error {model_key} idx {future2idx.get(future, '?')}: {e}")
                    pred_by_idx[future2idx[future]] = ""

    neds = [
        normalized_edit_distance(rows[i][2] if len(rows[i]) > 2 else "", pred_by_idx.get(i, ""))
        for i in range(len(rows))
    ]
    preds = [pred_by_idx.get(i, "") for i in range(len(rows))]
    mean_ned = sum(neds) / len(neds) if neds else 0.0
    return mean_ned, neds, preds


def main():
    parser = argparse.ArgumentParser(description="Compare LLMs on TKPD keyword extraction via NED (OpenRouter).")
    _repo_root = Path(__file__).resolve().parent.parent.parent.parent
    parser.add_argument(
        "--csv",
        default=str(_repo_root / "dataset" / "TKPD.csv"),
        help="Path to TKPD.csv",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini", "llama-3.1-8b", "qwen-2.5-7b", "deepseek", "llama-3.3-70b"],
        choices=list(OPENROUTER_MODELS),
        help="Model keys to compare",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows per model (0 = all)")
    parser.add_argument("--batch-size", type=int, default=8, help="Concurrent requests per batch (ThreadPoolExecutor max_workers)")
    parser.add_argument("--no-verbose", action="store_true", help="Less per-row output")
    parser.add_argument(
        "--out-dir",
        "-o",
        default=None,
        help="Save per-model CSV (title, abstract, field_gt, pred, ned) and ned_summary.json here",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), csv_path))

    print(f"Loading {csv_path} ...")
    rows = load_rows(csv_path)
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"Rows: {len(rows)}")

    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: Set OPENROUTER_API_KEY (or OPENAI_API_KEY) for OpenRouter.")

    results: dict[str, float] = {}
    all_preds: dict[str, list[str]] = {}
    all_neds: dict[str, list[float]] = {}

    for key in args.models:
        model_id = OPENROUTER_MODELS[key]
        print(f"\n--- {key} ({model_id}) ---")
        mean_ned, neds, preds = run_ned_for_model(
            rows, key, model_id, batch_size=args.batch_size, verbose=not args.no_verbose
        )
        results[key] = mean_ned
        all_neds[key] = neds
        all_preds[key] = preds
        print(f"Mean NED: {mean_ned:.4f}")

    print("\n========== NED comparison ==========")
    for key, mean_ned in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {key}: {mean_ned:.4f}")
    print("====================================\n")

    if args.out_dir:
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for key in args.models:
            csv_path = out_path / f"ned_results_{key}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["title", "abstract", "field_gt", "pred", "ned"])
                for i, row in enumerate(rows):
                    title = row[0]
                    abstract = row[1] if len(row) > 1 else ""
                    field_gt = row[2] if len(row) > 2 else ""
                    pred = all_preds[key][i]
                    ned = all_neds[key][i]
                    w.writerow([title, abstract, field_gt, pred, f"{ned:.6f}"])
            print(f"Wrote {csv_path}")
        summary_path = out_path / "ned_summary.json"
        summary = {
            "mean_ned_by_model": results,
            "per_model": {
                key: {
                    "mean_ned": results[key],
                    "neds": [round(n, 6) for n in all_neds[key]],
                    "preds": all_preds[key],
                }
                for key in args.models
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
