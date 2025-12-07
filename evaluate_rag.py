
#!/usr/bin/env python3
"""
Evaluate multiple models on test data, compare performance per subcategory and category,
with and without Retrieval-Augmented Generation (RAG), and visualize results.

Outputs:
- metrics/cls_report_*.json : classification reports per model & RAG flag
- metrics/text_metrics_*.json : text similarity metrics (BLEU, ROUGE-1, ROUGE-L, BERTScore) per model & RAG flag
- metrics/agg_metrics.csv : aggregated table per model/RAG/category/subcategory
- charts/*.png : visualization files

Requirements:
- sentence-transformers, qdrant-client (for RAG), nltk, rouge-score, bert-score, scikit-learn, pandas, matplotlib
- Qdrant accessible when RAG=True
- Ollama installed locally to run the models
"""

import os
import json
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.metrics import classification_report

# Local helpers (embeddings + RAG search)
from embeddings import (
    retrieve_similar_comments,
    get_client,
    load_model,
    ensure_collection,
    bulk_upsert_from_json,
    DEFAULT_COLLECTION,
)

# --------------------------- Configuration ---------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OUT_DIR = Path(os.getenv("OUT_DIR", "out_eval"))
METRICS_DIR = OUT_DIR / "metrics"
CHARTS_DIR = OUT_DIR / "charts"
for d in [OUT_DIR, METRICS_DIR, CHARTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATA_TEST = os.getenv("DATA_TEST", "data_splits_subcat/test.json")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
TOP_K = int(os.getenv("TOP_K", "3"))

MODELS = [
    "codellama:13b",
    "codestral:22b",
    "qwen3-coder:30b",
    "mistral:7b-instruct",
    "deepseek-coder-33b",
    "phi3:latest",
]

CANDIDATE_LABELS = [
    "functional", "logical", "validation", "resource", "timing", "support issues", "interface",
    "solution approach", "alternate output", "code organization", "variable naming", "visual representation",
    "documentation", "design discussion", "question", "praise", "false positive"
]

SUB_TO_TOP = {
    "functional": "functional",
    "logical": "functional",
    "validation": "functional",
    "resource": "functional",
    "timing": "functional",
    "support issues": "functional",
    "interface": "functional",
    "solution approach": "refactoring",
    "alternate output": "refactoring",
    "code organization": "refactoring",
    "variable naming": "refactoring",
    "visual representation": "refactoring",
    "documentation": "documentation",
    "design discussion": "discussion",
    "question": "discussion",
    "praise": "discussion",
    "false positive": "false positive"
}

# --------------------------- Utilities ---------------------------
def load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return [json.loads(line) for line in f if line.strip()]


def _format_similar(similar: List[Dict]) -> str:
    if not similar:
        return ""
    lines = []
    for s in similar:
        c = s.get("comment") or ""
        sub = s.get("subcategory") or ""
        cat = s.get("category") or ""
        sc = s.get("score")
        if sc is None:
            lines.append(f"- [{cat}/{sub}] {c}")
        else:
            lines.append(f"- [{cat}/{sub}] {c} (score={sc:.4f})")
    return "".join(lines)


def generate_prompt(code_snippet: str, similar_comments: List[Dict] | None = None) -> str:
    similar_block = _format_similar(similar_comments or [])
    ref_section = f"### Reference Comments (for similar code blocks):{similar_block} if similar_block else "
    prompt = f"""You are a senior Python developer tasked with reviewing the following pull request code.
Your goal is to provide a **technical, constructive, and actionable** review comment that is clear and helpful to developers.
---
### Code to Review:
{code_snippet}
{ref_section}---
### Instructions:
1. Analyze the code snippet carefully.
2. Assign a subcategory to your review, choosing exactly one from:
{CANDIDATE_LABELS}

### Output Format:
- **Review Comment:** <your detailed feedback here>
- **Subcategory:** <one of the 17 subcategories>
"""
    return prompt


async def run_ollama(prompt: str, model_name: str) -> str:
    try:
        process = await asyncio.create_subprocess_exec(
            "ollama", "run", model_name,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=prompt.encode("utf-8"))
        await asyncio.sleep(0.5)
        if process.returncode != 0:
            return f"Ollama error: {stderr.decode('utf-8')}"
        return stdout.decode("utf-8").strip()
    except Exception as e:
        return f"Exception: {str(e)}"


def parse_output(output: str) -> Tuple[str, str, str]:
    """Return (comment, predicted_subcategory, predicted_top_category)."""
    comment_match = re.search(r"Review\s*Comment:\s*(.*?)(?:-?\s*\*?\*?Subcategory|Subcategory:|$)",
        output,
        re.DOTALL,
    )
    subcategory_match = re.search(r"Subcategory:\s*(.+)", output)

    comment = comment_match.group(1).strip() if comment_match else output.strip()
    predicted_subcategory = (subcategory_match.group(1).strip().lower()
                             if subcategory_match else "unknown")
    predicted_top_category = SUB_TO_TOP.get(predicted_subcategory, "unknown")
    return comment, predicted_subcategory, predicted_top_category


def compute_text_metrics(references: List[str], predictions: List[str]) -> Dict[str, float]:
    bleu_scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=SmoothingFunction().method1)
        for ref, pred in zip(references, predictions)
    ]
    avg_bleu = float(sum(bleu_scores) / len(bleu_scores)) if bleu_scores else 0.0

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores, rougeL_scores = [], []
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    avg_rouge1 = float(sum(rouge1_scores) / len(rouge1_scores)) if rouge1_scores else 0.0
    avg_rougeL = float(sum(rougeL_scores) / len(rougeL_scores)) if rougeL_scores else 0.0

    try:
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        avg_bertscore_f1 = float(F1.mean().item())
    except Exception:
        avg_bertscore_f1 = 0.0

    return {
        "BLEU": avg_bleu,
        "ROUGE1_F1": avg_rouge1,
        "ROUGEL_F1": avg_rougeL,
        "BERTScore_F1": avg_bertscore_f1,
    }

# --------------------------- Core Evaluation ---------------------------
async def eval_model_on_test(model_name: str, use_rag: bool, items: List[Dict[str, Any]],
                             collection_name: str, top_k: int) -> pd.DataFrame:
    async def process_item(idx: int, item: Dict[str, Any]):
        code = item.get("code", "")
        similar = None
        if use_rag:
            try:
                similar = retrieve_similar_comments(code_snippet=code, top_k=top_k, collection_name=collection_name)
            except Exception:
                similar = None
        prompt = generate_prompt(code, similar)
        output = await run_ollama(prompt, model_name)
        comment_pred, sub_pred, cat_pred = parse_output(output)
        return {
            "idx": idx,
            "model": model_name,
            "use_rag": use_rag,
            "pred_comment": comment_pred,
            "pred_subcategory": sub_pred,
            "pred_category": cat_pred,
            "true_comment": item.get("comment", ""),
            "true_subcategory": str(item.get("subcategory", "")).strip().lower(),
            "true_category": str(item.get("category", "")).strip().lower() or SUB_TO_TOP.get(str(item.get("subcategory", "")).strip().lower(), "")
        }

    tasks = [process_item(i, it) for i, it in enumerate(items)]
    rows = await asyncio.gather(*tasks)
    return pd.DataFrame(rows)


def per_subcategory_class_report(df: pd.DataFrame) -> Dict[str, Any]:
    y_true = df["true_subcategory"].values
    y_pred = df["pred_subcategory"].values
    # classification_report with dict output includes per-class metrics
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def summarize_text_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    # overall
    overall = compute_text_metrics(df["true_comment"].tolist(), df["pred_comment"].tolist())
    # per subcategory
    by_sub = {}
    for sub, g in df.groupby("true_subcategory"):
        by_sub[sub] = compute_text_metrics(g["true_comment"].tolist(), g["pred_comment"].tolist())
    # per category
    by_cat = {}
    for cat, g in df.groupby("true_category"):
        by_cat[cat] = compute_text_metrics(g["true_comment"].tolist(), g["pred_comment"].tolist())
    return {"overall": overall, "per_subcategory": by_sub, "per_category": by_cat}


def to_agg_rows(model: str, use_rag: bool, cls_rep: Dict[str, Any], text_met: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    # overall
    rows.append({
        "model": model,
        "use_rag": use_rag,
        "scope": "overall",
        "label": "ALL",
        "f1": cls_rep.get("macro avg", {}).get("f1-score", None),
        "precision": cls_rep.get("macro avg", {}).get("precision", None),
        "recall": cls_rep.get("macro avg", {}).get("recall", None),
        "BLEU": text_met["overall"].get("BLEU"),
        "ROUGE1_F1": text_met["overall"].get("ROUGE1_F1"),
        "ROUGEL_F1": text_met["overall"].get("ROUGEL_F1"),
        "BERTScore_F1": text_met["overall"].get("BERTScore_F1"),
    })
    # per subcategory
    for sub, stats in cls_rep.items():
        if sub in ["accuracy", "macro avg", "weighted avg"]:
            continue
        rows.append({
            "model": model,
            "use_rag": use_rag,
            "scope": "subcategory",
            "label": sub,
            "f1": stats.get("f1-score", None),
            "precision": stats.get("precision", None),
            "recall": stats.get("recall", None),
            "BLEU": text_met["per_subcategory"].get(sub, {}).get("BLEU"),
            "ROUGE1_F1": text_met["per_subcategory"].get(sub, {}).get("ROUGE1_F1"),
            "ROUGEL_F1": text_met["per_subcategory"].get(sub, {}).get("ROUGEL_F1"),
            "BERTScore_F1": text_met["per_subcategory"].get(sub, {}).get("BERTScore_F1"),
        })
    # per category
    for cat, stats in text_met["per_category"].items():
        # Build classification numbers by aggregating subcategories mapped to this category
        # Use macro avg over those labels from cls_report
        subs = [s for s, c in SUB_TO_TOP.items() if c == cat]
        f1_vals = []
        prec_vals = []
        rec_vals = []
        for s in subs:
            if s in cls_rep:
                f1_vals.append(cls_rep[s].get("f1-score", np.nan))
                prec_vals.append(cls_rep[s].get("precision", np.nan))
                rec_vals.append(cls_rep[s].get("recall", np.nan))
        rows.append({
            "model": model,
            "use_rag": use_rag,
            "scope": "category",
            "label": cat,
            "f1": np.nanmean(f1_vals) if f1_vals else np.nan,
            "precision": np.nanmean(prec_vals) if prec_vals else np.nan,
            "recall": np.nanmean(rec_vals) if rec_vals else np.nan,
            "BLEU": stats.get("BLEU"),
            "ROUGE1_F1": stats.get("ROUGE1_F1"),
            "ROUGEL_F1": stats.get("ROUGEL_F1"),
            "BERTScore_F1": stats.get("BERTScore_F1"),
        })
    return rows


def plot_grouped_bars(df: pd.DataFrame, title: str, filename: Path, metric: str = "f1",
                      hue: str = "use_rag", xlabel: str = "label"):
    # df columns: label, use_rag, f1, model
    plt.figure(figsize=(14, 7))
    # Pivot to have RAG vs Non-RAG
    pivot = df.pivot_table(index=xlabel, columns=hue, values=metric)
    pivot.plot(kind='bar')
    plt.title(title)
    plt.ylabel(metric.upper())
    plt.xlabel(xlabel)
    plt.legend(title=hue)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


async def main():
    items = load_data(DATA_TEST)
    all_agg_rows = []

    for model in MODELS:
        # Evaluate with RAG
        df_rag = await eval_model_on_test(model_name=model, use_rag=True, items=items,
                                          collection_name=COLLECTION_NAME, top_k=TOP_K)
        # Evaluate without RAG
        df_no = await eval_model_on_test(model_name=model, use_rag=False, items=items,
                                         collection_name=COLLECTION_NAME, top_k=TOP_K)

        # Classification reports
        cls_rag = per_subcategory_class_report(df_rag)
        cls_no = per_subcategory_class_report(df_no)
        # Text metrics
        txt_rag = summarize_text_metrics(df_rag)
        txt_no = summarize_text_metrics(df_no)

        # Save metrics
        with open(METRICS_DIR / f"cls_report_{model}_rag.json", "w", encoding="utf-8") as f:
            json.dump(cls_rag, f, indent=2)
        with open(METRICS_DIR / f"cls_report_{model}_no_rag.json", "w", encoding="utf-8") as f:
            json.dump(cls_no, f, indent=2)
        with open(METRICS_DIR / f"text_metrics_{model}_rag.json", "w", encoding="utf-8") as f:
            json.dump(txt_rag, f, indent=2)
        with open(METRICS_DIR / f"text_metrics_{model}_no_rag.json", "w", encoding="utf-8") as f:
            json.dump(txt_no, f, indent=2)

        # Aggregate rows for comparison table
        all_agg_rows.extend(to_agg_rows(model, True, cls_rag, txt_rag))
        all_agg_rows.extend(to_agg_rows(model, False, cls_no, txt_no))

        # Visualizations per model
        # Subcategory F1 grouped bars: RAG vs Non-RAG
        sub_df = pd.DataFrame([r for r in all_agg_rows if r["model"] == model and r["scope"] == "subcategory"])
        if not sub_df.empty:
            plot_grouped_bars(
                df=sub_df[["label", "use_rag", "f1"]],
                title=f"Per-Subcategory F1: {model} (RAG vs Non-RAG)",
                filename=CHARTS_DIR / f"{model}_subcategory_f1.png",
            )
        # Category F1 grouped bars
        cat_df = pd.DataFrame([r for r in all_agg_rows if r["model"] == model and r["scope"] == "category"])
        if not cat_df.empty:
            plot_grouped_bars(
                df=cat_df[["label", "use_rag", "f1"]],
                title=f"Per-Category F1: {model} (RAG vs Non-RAG)",
                filename=CHARTS_DIR / f"{model}_category_f1.png",
            )
        # Overall F1 grouped bars
        overall_df = pd.DataFrame([r for r in all_agg_rows if r["model"] == model and r["scope"] == "overall"])
        if not overall_df.empty:
            plt.figure(figsize=(6, 4))
            overall_df.set_index("use_rag")["f1"].plot(kind='bar', color=['C0', 'C1'])
            plt.title(f"Overall Macro-F1: {model} (RAG vs Non-RAG)")
            plt.ylabel("F1")
            plt.xlabel("use_rag")
            plt.tight_layout()
            plt.savefig(CHARTS_DIR / f"{model}_overall_f1.png")
            plt.close()

    # Save aggregated comparison table
    agg_df = pd.DataFrame(all_agg_rows)
    agg_df.to_csv(METRICS_DIR / "agg_metrics.csv", index=False)

    # Global comparison across models: overall F1 RAG vs Non-RAG
    overall_all = agg_df[agg_df["scope"] == "overall"]
    if not overall_all.empty:
        plt.figure(figsize=(10, 6))
        # Create grouped bars by model with hue use_rag
        pivot = overall_all.pivot_table(index="model", columns="use_rag", values="f1")
        pivot.plot(kind='bar')
        plt.title("Overall Macro-F1 by Model: RAG vs Non-RAG")
        plt.ylabel("F1")
        plt.xlabel("Model")
        plt.legend(title="use_rag")
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "overall_models_f1.png")
        plt.close()

if __name__ == "__main__":
    asyncio.run(main())
