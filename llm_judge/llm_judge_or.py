import pandas as pd
import json
import re
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
# INPUT_FILE = "/NAIDv1/NAID_test_extrainfo.csv"
# INPUT_FILE = "/NAIDv2/NAIDv2-test.csv"

# OUTPUT_JSON = "/ver1/test/naidv1/Llama-3.1/llm_scores.json"     
# OUTPUT_JSON = "/mnt/data/son/Reviewerly/dataset/ver1/test/naidv2/Llama-3.1/llm_scores.json" 

# INPUT_FILE = "/NAIDv1/NAID_train_extrainfo.csv"
INPUT_FILE = "/NAIDv2/NAIDv2-train.csv"

# OUTPUT_JSON = "/ver1/train/naidv1/Llama-3.1/llm_scores.json"     
OUTPUT_JSON = "/ver1/train/naidv2/Llama-3.1/llm_scores.json"      


# MODEL_NAME = "deepseek/deepseek-chat-v3-0324:floor"
# MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"
# MODEL_NAME = "openai/gpt-4.1"

API_BASE = "https://openrouter.ai/api/v1"
API_KEY = "                 

BATCH_SIZE = 2000  # Number of parallel/concurrent requests

# The 60 Items from Table 1 of the paper [cite: 86, 87]
ITEMS = [
    "1. Engaging", "2. Controversial", "3. Rigorous", "4. Innovative", "5. Accessible",
    "6. Methodical", "7. Concise", "8. Persuasive", "9. Comprehensive", "10. Insightful",
    "11. Relevant", "12. Objective", "13. Replicable", "14. Structured", "15. Coherent",
    "16. Original", "17. Balanced", "18. Authoritative", "19. Impactful", "20. Interdisciplinary",
    "21. Well-sourced", "22. Technical", "23. Provocative", "24. Hypothesis-driven", "25. Ethical",
    "26. Difficult to understand", "27. Exciting", "28. Not well written", "29. Theoretical", "30. To the point",
    # Antonyms (Items 31-60)
    "31. Disengaging", "32. Uncontroversial", "33. Lax", "34. Conventional", "35. Inaccessible",
    "36. Haphazard", "37. Verbose", "38. Unconvincing", "39. Superficial", "40. Uninsightful",
    "41. Irrelevant", "42. Subjective", "43. Non-replicable", "44. Unstructured", "45. Incoherent",
    "46. Derivative", "47. Unbalanced", "48. Unreliable", "49. Inconsequential", "50. Narrow",
    "51. Poorly-sourced", "52. Nontechnical", "53. Unprovocative", "54. Speculation-driven", "55. Unethical",
    "56. Easy to understand", "57. Dull", "58. Well written", "59. Empirical", "60. Circumlocutory"
]



# ITEMS = [
#     "1. Circumlocutory", "2. Empirical", "3. Well written", "4. Dull", "5. Easy to understand",
#     "6. Unethical", "7. Speculation-driven", "8. Unprovocative", "9. Nontechnical", "10. Poorly-sourced",
#     "11. Narrow", "12. Inconsequential", "13. Unreliable", "14. Unbalanced", "15. Derivative",
#     "16. Incoherent", "17. Unstructured", "18. Non-replicable", "19. Subjective", "20. Irrelevant",
#     "21. Uninsightful", "22. Superficial", "23. Unconvincing", "24. Verbose", "25. Haphazard",
#     "26. Inaccessible", "27. Conventional", "28. Lax", "29. Uncontroversial", "30. Disengaging",
#     "31. To the point", "32. Theoretical", "33. Not well written", "34. Exciting", "35. Difficult to understand",
#     "36. Ethical", "37. Hypothesis-driven", "38. Provocative", "39. Technical", "40. Well-sourced",
#     "41. Interdisciplinary", "42. Impactful", "43. Authoritative", "44. Balanced", "45. Original",
#     "46. Coherent", "47. Structured", "48. Replicable", "49. Objective", "50. Relevant",
#     "51. Insightful", "52. Comprehensive", "53. Persuasive", "54. Concise", "55. Methodical",
#     "56. Accessible", "57. Innovative", "58. Rigorous", "59. Controversial", "60. Engaging"
# ]


# ITEMS = [
#     "1. Insightful", "2. Haphazard", "3. Well-sourced", "4. Disengaging", "5. Technical",
#     "6. Balanced", "7. Superficial", "8. Ethical", "9. Concise", "10. Unreliable",
#     "11. Innovative", "12. Subjective", "13. Circumlocutory", "14. Rigorous", "15. Dull",
#     "16. Replicable", "17. Provocative", "18. Inaccessible", "19. Empirical", "20. Coherent",
#     "21. Poorly-sourced", "22. Impactful", "23. Nontechnical", "24. Unstructured", "25. Engaging",
#     "26. Narrow", "27. Methodical", "28. Unconvincing", "29. Exciting", "30. Speculation-driven",
#     "31. Relevant", "32. Lax", "33. Well written", "34. Hypothesis-driven", "35. Inconsequential",
#     "36. Accessible", "37. Authoritative", "38. Not well written", "39. Interdisciplinary",
#     "40. Easy to understand", "41. Original", "42. Unbalanced", "43. Comprehensive",
#     "44. Verbose", "45. Conventional", "46. To the point", "47. Unprovocative",
#     "48. Structured", "49. Difficult to understand", "50. Irrelevant",
#     "51. Controversial", "52. Non-replicable", "53. Objective", "54. Derivative",
#     "55. Unethical", "56. Persuasive", "57. Incoherent", "58. Uninsightful",
#     "59. Theoretical", "60. Exciting"
# ]


def clean_abstract(text):
    """Removes copyright notices as done in the study[cite: 60, 61]."""
    if not isinstance(text, str): return ""
    text = re.sub(r'Copyright © \d{4}.*', '', text, flags=re.IGNORECASE)
    return text.strip()

def build_prompt(title, abstract):
    """Create the prompt for the LLM."""
    items_str = "\n".join(ITEMS)
    # random.seed(42)
    # random.shuffle(ITEMS)
    prompt = f"""Please rate the following abstract on each of the 60 items from 0=Not at all to 100=Very much. Only provide the numbers. For example:
1.65
2.50
3.5
4.95
5....

This is the title and abstract:
"{title} {abstract}"

These are the items:
{items_str}
"""
    return prompt

def extract_scores(response_content):
    # Extract all numbers (integers or floats) from the response
    scores = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", response_content)]
    # We expect exactly 60 scores. If we get more (e.g., line numbers), take the last 60 or filter logic.
    return scores[:60]

def query_llm(client, row_dict):
    """Query the LLM with retry logic. Return (record dict or None, error)."""
    title_text = clean_abstract(row_dict.get('title', ''))
    abstract_text = clean_abstract(row_dict.get('abstract', ''))
    if not abstract_text:
        return None, "Empty abstract"

    prompt = build_prompt(title_text, abstract_text)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # For reproducibility
            max_tokens=1000
        )
        content = completion.choices[0].message.content
        scores = extract_scores(content)
    except Exception as e:
        return None, f"Exception: {e}"

    if len(scores) != 60:
        # Retry ONCE
        try:
            completion2 = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                # max_tokens=1000
            )
            content2 = completion2.choices[0].message.content
            scores = extract_scores(content2)
            content = content2
        except Exception as e:
            return None, f"Exception on retry: {e}"

    if len(scores) == 60:
        rec = {
            "id": row_dict.get('id', row_dict.get('idx', None)),
            "cites": row_dict.get('cites', 0), # Target variable for analysis
            "title": row_dict.get('title', ''),
            "raw_scores": scores,
            "llm_response": content
        }
        return rec, None
    else:
        return None, f"Failed: Only {len(scores)} scores after retry"

def main():
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} abstracts.")

    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    results = []
    skip_counter = 0
    records = df.to_dict(orient="records")[:2000]

    for start in tqdm(range(0, len(records), BATCH_SIZE)):
        batch = records[start : start + BATCH_SIZE]

        # Launch in a thread pool for concurrent throughput (adjust max_workers if API/server supports more/less)
        batch_results = []
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            # Map index (to preserve order if needed)
            future2row = {
                executor.submit(query_llm, client, row): row
                for row in batch
            }
            for future in as_completed(future2row):
                row = future2row[future]
                try:
                    record, error = future.result()
                except Exception as exc:
                    print(f"Exception for row {row.get('id', row.get('idx', 'UNKNOWN'))}: {exc}")
                    record = None
                    error = f"raise: {exc}"

                if record is not None:
                    results.append(record)
                else:
                    skip_counter += 1
                    print(f"Skipping ID {row.get('id', row.get('idx', 'UNKNOWN'))}: {error}")
        # break

    # 3. Save to JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} scored abstracts to {OUTPUT_JSON}, skipped {skip_counter}")

if __name__ == "__main__":
    main()