import argparse
import csv
import json
import os
import signal
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from lingua import Language, LanguageDetectorBuilder, IsoCode639_1

sys.path.append(str(Path(__file__).resolve().parent / "pyscripts"))

from multilingual_utils import normalize_lang, get_smile_emb_model, unicode_normalize, SUPPORTED_LANGUAGES
from smile.smile import SMILE
from utils import (
    compute_rouge_score, compute_bert_score, compute_meteor_score,
    compute_exact_match, compute_sbert_score, compute_bleurt_score,
    compute_moverscore,
)


# ─── Language detection helpers ───

_LANG_TO_LINGUA = {
    "ar": Language.ARABIC,
    "bn": Language.BENGALI,
    "en": Language.ENGLISH,
    "fi": Language.FINNISH,
    "ja": Language.JAPANESE,
    "ko": Language.KOREAN,
    "ru": Language.RUSSIAN,
    "te": Language.TELUGU,
}
_LINGUA_TO_LANG = {v: k for k, v in _LANG_TO_LINGUA.items()}
_detector_cache = {}


def _get_detector(question_lang: str):
    """Get or build a lingua detector that chooses between English and question_lang."""
    key = question_lang
    if key not in _detector_cache:
        langs = [Language.ENGLISH]
        q_lingua = _LANG_TO_LINGUA.get(question_lang)
        if q_lingua and q_lingua != Language.ENGLISH:
            langs.append(q_lingua)
        _detector_cache[key] = LanguageDetectorBuilder.from_languages(*langs).build()
    return _detector_cache[key]


def detect_answer_language(answer_text: str, question_lang: str) -> str:
    """Detect whether the answer is in English or the question's language."""
    question_lang = normalize_lang(question_lang)
    if question_lang == "en":
        return "en"
    if not answer_text or not answer_text.strip():
        return question_lang
    detector = _get_detector(question_lang)
    detected = detector.detect_language_of(answer_text)
    if detected is None:
        return question_lang
    return _LINGUA_TO_LANG.get(detected, question_lang)


# ─── Global state for signal handler ───
_checkpoint_state = {
    "path": None,
    "results_per_question": None,
    "completed_tasks": None,
}


def _save_checkpoint():
    """Save current progress to checkpoint file."""
    path = _checkpoint_state["path"]
    if path is None:
        return
    data = {
        "results_per_question": _checkpoint_state["results_per_question"],
        "completed_tasks": list(_checkpoint_state["completed_tasks"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
    print(f"\n  [CHECKPOINT] Saved to {path} ({len(data['completed_tasks'])} tasks done)", flush=True)


def _load_checkpoint(path):
    """Load checkpoint if it exists. Returns (results_per_question, completed_tasks) or (None, None)."""
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data["results_per_question"]
        completed = set(tuple(t) for t in data["completed_tasks"])
        print(f"  [CHECKPOINT] Resumed from {path} — {len(completed)} tasks already done (saved {data.get('timestamp', '?')})", flush=True)
        return results, completed
    except Exception as e:
        print(f"  [CHECKPOINT] Failed to load {path}: {e}. Starting fresh.", flush=True)
        return None, None


def _sigusr1_handler(signum, frame):
    """Handle SIGUSR1 from SLURM (preemption warning). Save checkpoint and exit."""
    print(f"\n{'='*70}", flush=True)
    print("[SIGNAL] Received SIGUSR1 — SLURM preemption imminent.", flush=True)
    print("Saving checkpoint and exiting gracefully...", flush=True)
    print(f"{'='*70}", flush=True)
    _save_checkpoint()
    sys.exit(42)


signal.signal(signal.SIGUSR1, _sigusr1_handler)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multilingual SMILE Metric Evaluation (GPU with checkpointing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input JSONL file with LLM answers.")
    parser.add_argument("--ground-truth", type=str, required=True,
                        help="Path to the ground-truth JSONL file.")
    parser.add_argument("--output", type=str, default="evaluation_results.csv",
                        help="Path to the output file (.csv or .jsonl)")
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["smile", "rouge", "bertscore", "meteor", "exact_match", "sbert", "bleurt", "moverscore"],
                        help="Which metrics to compute")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress messages")
    parser.add_argument("--smile-batch-size", type=int, default=512,
                        help="Batch size for SMILE generate_scores calls (default: 512)")
    parser.add_argument("--metric-workers", type=int, default=3,
                        help="Number of parallel workers for non-SMILE metrics (default: 3)")
    return parser.parse_args()


def load_jsonl(filepath: str) -> list:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line {line_num} in {filepath}: {e}")
    return data


def merge_data(input_data: list, gt_data: list) -> list:
    """Merge input (LLM answers) with ground-truth by question_id.

    Detects the actual language of the first reference answer using lingua
    (binary detection between English and the question language).
    """
    gt_lookup = {}
    for item in gt_data:
        qid = str(item.get("question_id", item.get("id", "")))
        gt_lookup[qid] = item

    merged = []
    missing_count = 0
    lang_detect_stats = {"same": 0, "english": 0}

    for item in input_data:
        qid = str(item.get("question_id", item.get("id", "")))
        gt = gt_lookup.get(qid)
        if gt is None:
            missing_count += 1
            continue

        answers = gt.get("answers", gt.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]

        question_lang = gt.get("lang", "en")

        # Detect the language of the first reference answer
        first_answer = answers[0] if answers else ""
        detected_lang = detect_answer_language(first_answer, question_lang)

        if detected_lang == "en" and question_lang != "en":
            lang_detect_stats["english"] += 1
        else:
            lang_detect_stats["same"] += 1

        merged.append({
            "question_id": qid,
            "question": item.get("question", ""),
            "pred": item.get("answer", ""),
            "answers": answers,
            "lang": detected_lang,
            "question_lang": question_lang,
        })

    if missing_count > 0:
        print(f"WARNING: {missing_count} questions in input had no matching ground-truth (skipped)")

    print(f"  Language detection: {lang_detect_stats['same']} answers in question language, "
          f"{lang_detect_stats['english']} answers detected as English")

    return merged


def group_by_language(merged_data: list) -> dict:
    """Group merged data by language for efficient batch processing."""
    groups = defaultdict(list)
    for item in merged_data:
        lang = normalize_lang(item["lang"])
        groups[lang].append(item)
    return dict(groups)


def compute_smile_scores(items: list, lang: str, batch_size: int = 512, verbose: bool = False) -> list:
    """
    Compute SMILE scores for a batch of items in the same language.

    KEY OPTIMIZATION: Instead of calling generate_scores once per item (61k calls),
    we expand all (item, reference) pairs into a single large qa_set and call
    generate_scores in large batches. This amortizes model overhead and allows
    GPU to process full batches rather than single rows.
    """
    emb_model = get_smile_emb_model(lang)
    eval_metrics = ['avg', 'hm']
    smile_obj = SMILE(
        emb_model=emb_model,
        eval_metrics=eval_metrics,
        lang=lang,
        assign_bins=False,
        use_exact_matching=True,
        verbose=verbose,
    )

    # Expand: each item may have multiple reference answers.
    # Track which original item each row belongs to.
    all_rows = []       # list of [question, ref, ref, pred]
    row_to_item = []    # parallel list: which item index does this row belong to

    for item_idx, item in enumerate(items):
        for ref_ans in item["answers"]:
            all_rows.append([item["question"], ref_ans, ref_ans, item["pred"]])
            row_to_item.append(item_idx)

    # Run generate_scores in batches over the expanded rows
    all_avg = np.zeros(len(all_rows))
    all_hm  = np.zeros(len(all_rows))

    total_batches = (len(all_rows) + batch_size - 1) // batch_size
    for batch_idx in range(0, len(all_rows), batch_size):
        batch_rows = all_rows[batch_idx: batch_idx + batch_size]
        qa_set = np.array(batch_rows)
        current_batch = batch_idx // batch_size + 1
        if verbose or (current_batch % 10 == 0):
            print(f"    SMILE batch {current_batch}/{total_batches} "
                  f"(rows {batch_idx}–{batch_idx + len(batch_rows) - 1})", flush=True)
        try:
            results = smile_obj.generate_scores(qa_set)
            all_avg[batch_idx: batch_idx + len(batch_rows)] = results["avg"]
            all_hm [batch_idx: batch_idx + len(batch_rows)] = results["hm"]
        except Exception as e:
            if verbose:
                print(f"  SMILE batch error at rows {batch_idx}-{batch_idx+len(batch_rows)-1}: {e}")
            # scores stay 0 for this batch

    # Aggregate: take max across references for each item
    best_avg = np.zeros(len(items))
    best_hm  = np.zeros(len(items))
    for row_idx, item_idx in enumerate(row_to_item):
        if all_avg[row_idx] > best_avg[item_idx]:
            best_avg[item_idx] = all_avg[row_idx]
        if all_hm[row_idx] > best_hm[item_idx]:
            best_hm[item_idx] = all_hm[row_idx]

    return [{"smile_avg": float(best_avg[i]), "smile_hm": float(best_hm[i])}
            for i in range(len(items))]


def run_batched_metric(items: list, lang: str, metric_name: str, verbose: bool = False) -> list:
    """
    Batch computation for a metric, supporting multiple reference answers.

    KEY OPTIMIZATION: For multi-reference items, we expand all (item, ref) pairs
    into one flat batch, run the metric once, then take the max per item.
    This replaces the old per-item loop that ran the metric N_refs times per item.
    """
    key_map = {
        "rouge":        ("rougeL", "rougeL", "fmeasure"),
        "bertscore":    ("F1",     None,     None),
        "meteor":       ("meteor", "meteor", None),
        "exact_match":  ("exact_match", "exact_match", None),
        "sbert":        (None,     None,     None),
        "bleurt":       ("scores", "scores", None),
        "moverscore":   ("scores", "scores", None),
    }

    # Build flat list of rows and track item origin
    flat_rows = []
    row_to_item = []
    for item_idx, item in enumerate(items):
        for ref_ans in item["answers"]:
            flat_rows.append((item["question"], ref_ans, ref_ans, item["pred"]))
            row_to_item.append(item_idx)

    try:
        if metric_name == "rouge":
            result = compute_rouge_score(metrics=['rougeL'], ref_data=flat_rows,
                                         sub_metrics=['fmeasure'], ans_idx=1, lang=lang)
            flat_scores = result['rougeL']['fmeasure']
        elif metric_name == "bertscore":
            result = compute_bert_score(inp_data=flat_rows, ans_idx=1, lang=lang)
            flat_scores = result['F1']
        elif metric_name == "meteor":
            result = compute_meteor_score(inp_data=flat_rows, ans_idx=1, lang=lang)
            flat_scores = result['meteor']
        elif metric_name == "exact_match":
            result = compute_exact_match(inp_data=flat_rows, ans_idx=1, lang=lang)
            flat_scores = result['exact_match']
        elif metric_name == "sbert":
            result = compute_sbert_score(inp_data=flat_rows, ans_idx=1, lang=lang)
            flat_scores = result.tolist() if hasattr(result, 'tolist') else list(result)
        elif metric_name == "bleurt":
            result = compute_bleurt_score(inp_data=flat_rows, ans_idx=1, lang=lang)
            flat_scores = result['scores']
        elif metric_name == "moverscore":
            device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
            result = compute_moverscore(inp_data=flat_rows, ans_idx=1, lang=lang, device=device)
            flat_scores = result['scores']
        else:
            flat_scores = [0.0] * len(flat_rows)

    except Exception as e:
        if verbose:
            print(f"  Batch {metric_name} failed: {e}. Falling back to per-item.")
        # Per-item fallback (original slow path)
        flat_scores = []
        for row in flat_rows:
            try:
                if metric_name == "rouge":
                    r = compute_rouge_score(metrics=['rougeL'], ref_data=[row],
                                            sub_metrics=['fmeasure'], ans_idx=1, lang=lang)
                    flat_scores.append(r['rougeL']['fmeasure'][0])
                elif metric_name == "bertscore":
                    r = compute_bert_score(inp_data=[row], ans_idx=1, lang=lang)
                    flat_scores.append(r['F1'][0])
                elif metric_name == "meteor":
                    r = compute_meteor_score(inp_data=[row], ans_idx=1, lang=lang)
                    flat_scores.append(r['meteor'][0])
                elif metric_name == "exact_match":
                    r = compute_exact_match(inp_data=[row], ans_idx=1, lang=lang)
                    flat_scores.append(r['exact_match'][0])
                elif metric_name == "sbert":
                    r = compute_sbert_score(inp_data=[row], ans_idx=1, lang=lang)
                    flat_scores.append(float(r[0]))
                elif metric_name == "bleurt":
                    r = compute_bleurt_score(inp_data=[row], ans_idx=1, lang=lang)
                    flat_scores.append(r['scores'][0])
                elif metric_name == "moverscore":
                    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
                    r = compute_moverscore(inp_data=[row], ans_idx=1, lang=lang, device=device)
                    flat_scores.append(r['scores'][0])
                else:
                    flat_scores.append(0.0)
            except Exception as e2:
                if verbose:
                    print(f"    per-item {metric_name} error: {e2}")
                flat_scores.append(0.0)

    # Aggregate: max score across references per item
    best_scores = [-float("inf")] * len(items)
    for row_idx, item_idx in enumerate(row_to_item):
        s = float(flat_scores[row_idx])
        if s > best_scores[item_idx]:
            best_scores[item_idx] = s

    return [s if s > -float("inf") else 0.0 for s in best_scores]


def write_output(output_path, results_per_question, per_lang_summary, summary, metric_keys):
    """Write results to CSV or JSONL."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if output_path.lower().endswith('.csv'):
        fieldnames = ["question_id", "lang", "question_lang"] + metric_keys
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in results_per_question:
                writer.writerow(r)
            for lang_summary in per_lang_summary.values():
                writer.writerow(lang_summary)
            writer.writerow(summary)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results_per_question:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for lang_summary in per_lang_summary.values():
                f.write(json.dumps(lang_summary, ensure_ascii=False) + "\n")
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def main():
    args = parse_arguments()

    _checkpoint_state["output_path"] = args.output

    base_path = args.output
    if base_path.endswith('.csv'):
        base_path = base_path[:-4]

    checkpoint_path = base_path + ".checkpoint.json"
    _checkpoint_state["path"] = checkpoint_path

    if os.path.exists(args.output) and not os.path.exists(checkpoint_path):
        print(f"Output file {args.output} already exists and no checkpoint found.")
        print("Assuming this file is fully evaluated. Skipping.")
        return

    # ── 1. Load data ──
    print("Step 1: Loading input files", flush=True)
    input_data = load_jsonl(args.input)
    gt_data = load_jsonl(args.ground_truth)
    print(f"  Input:        {len(input_data)} questions")
    print(f"  Ground-truth: {len(gt_data)} questions")

    # ── 2. Merge ──
    print("\nStep 2: Merging by question_id", flush=True)
    merged = merge_data(input_data, gt_data)
    print(f"  Matched: {len(merged)} questions")

    if len(merged) == 0:
        print("ERROR: No matching question_ids found. Check your files.")
        sys.exit(1)

    # ── 3. Group by language ──
    lang_groups = group_by_language(merged)
    print("\n  Language distribution:")
    for lang, items in sorted(lang_groups.items()):
        print(f"    {lang}: {len(items)} questions")

    qid_to_idx = {item["question_id"]: i for i, item in enumerate(merged)}

    # ── 4. Load checkpoint or initialize ──
    results_per_question, completed_tasks = _load_checkpoint(checkpoint_path)
    if results_per_question is None:
        results_per_question = [{
            "question_id": item["question_id"],
            "lang": normalize_lang(item["lang"]),
            "question_lang": normalize_lang(item["question_lang"]),
        } for item in merged]
        completed_tasks = set()

    _checkpoint_state["results_per_question"] = results_per_question
    _checkpoint_state["completed_tasks"] = completed_tasks

    metrics_to_run = [m.lower() for m in args.metrics]
    total_start = time.time()

    display_names = {
        "rouge": "ROUGE-L", "bertscore": "BERTScore", "meteor": "METEOR",
        "exact_match": "Exact Match", "sbert": "sBERT",
        "bleurt": "BLEURT", "moverscore": "MoverScore",
    }
    key_map = {
        "rouge": "rouge_l", "bertscore": "bert_score_f1",
        "meteor": "meteor", "exact_match": "exact_match",
        "sbert": "sbert", "bleurt": "bleurt", "moverscore": "moverscore",
    }

    # ── 5. Compute metrics per language ──
    for lang, items in sorted(lang_groups.items()):
        indices = [qid_to_idx[item["question_id"]] for item in items]

        # ── SMILE (GPU-heavy; run alone) ──
        if "smile" in metrics_to_run:
            task_key = (lang, "smile")
            if task_key not in completed_tasks:
                print(f"\n  Computing SMILE for {lang} ({len(items)} questions)...", flush=True)
                t0 = time.time()
                smile_scores = compute_smile_scores(
                    items, lang,
                    batch_size=args.smile_batch_size,
                    verbose=args.verbose,
                )
                for idx, scores in zip(indices, smile_scores):
                    results_per_question[idx].update(scores)
                completed_tasks.add(task_key)
                _save_checkpoint()
                print(f"    Done in {time.time() - t0:.1f}s", flush=True)
            else:
                print(f"\n  [SKIP] SMILE for {lang} — already completed", flush=True)

        # ── Other metrics (run in parallel with ThreadPoolExecutor) ──
        other_metrics = [m for m in metrics_to_run if m != "smile"]
        pending = [m for m in other_metrics if (lang, m) not in completed_tasks]
        skipped = [m for m in other_metrics if (lang, m) in completed_tasks]

        for m in skipped:
            print(f"\n  [SKIP] {display_names.get(m, m)} for {lang} — already completed", flush=True)

        if not pending:
            continue

        print(f"\n  Computing {len(pending)} metrics for {lang} ({len(items)} questions) "
              f"with up to {args.metric_workers} workers...", flush=True)

        def _run_one_metric(metric_name):
            t0 = time.time()
            scores = run_batched_metric(items, lang, metric_name, verbose=args.verbose)
            elapsed = time.time() - t0
            return metric_name, scores, elapsed

        # Use threads — safe because each metric uses its own model/state,
        # and GIL is released during C-extension/GPU work (torch, transformers).
        with ThreadPoolExecutor(max_workers=args.metric_workers) as executor:
            futures = {executor.submit(_run_one_metric, m): m for m in pending}
            for future in as_completed(futures):
                metric_name, scores, elapsed = future.result()
                key = key_map.get(metric_name, metric_name)
                for idx, score in zip(indices, scores):
                    results_per_question[idx][key] = float(score)
                completed_tasks.add((lang, metric_name))
                _save_checkpoint()
                print(f"    {display_names.get(metric_name, metric_name)} done in {elapsed:.1f}s", flush=True)

    total_time = time.time() - total_start

    # ── 6. Aggregate summary ──
    print("\nComputing aggregate summary", flush=True)

    metric_keys = set()
    for r in results_per_question:
        for k in r:
            if k not in ("question_id", "lang", "question_lang"):
                metric_keys.add(k)
    metric_keys = sorted(metric_keys)

    summary = {"question_id": "AGGREGATE_SUMMARY", "lang": "all"}
    for key in metric_keys:
        values = [r[key] for r in results_per_question if key in r]
        if values:
            summary[key] = float(np.mean(values))

    # Per-language mean — grouped by question language
    question_lang_groups = defaultdict(list)
    for item in merged:
        question_lang_groups[normalize_lang(item["question_lang"])].append(item)

    per_lang_summary = {}
    for lang, items in sorted(question_lang_groups.items()):
        lang_indices = [qid_to_idx[item["question_id"]] for item in items]
        lang_summary = {"question_id": f"SUMMARY_{lang.upper()}", "lang": lang}
        for key in metric_keys:
            values = [results_per_question[i][key] for i in lang_indices if key in results_per_question[i]]
            if values:
                lang_summary[key] = float(np.mean(values))
        per_lang_summary[lang] = lang_summary

    # ── 7. Write output ──
    print(f"\nWriting results to {args.output}", flush=True)
    write_output(args.output, results_per_question, per_lang_summary, summary, metric_keys)

    # ── 8. Print summary ──
    print(f"\nEVALUATION COMPLETE — {len(merged)} questions, {total_time:.1f}s total", flush=True)

    print(f"\nOverall Scores (mean across all {len(merged)} questions):")
    print(f"  {'Metric':<20} {'Score':>8}")
    print(f"  {'-'*20} {'-'*8}")
    for key in metric_keys:
        if key in summary:
            print(f"  {key:<20} {summary[key]:>8.4f}")

    print(f"\nPer-Language Scores:")
    for lang, lang_summary in sorted(per_lang_summary.items()):
        n = len(question_lang_groups[lang])
        print(f"\n  {lang.upper()} ({n} questions):")
        for key in metric_keys:
            if key in lang_summary:
                print(f"    {key:<20} {lang_summary[key]:>8.4f}")

    print(f"\nResults written to: {args.output}")

    # ── 9. Clean up checkpoint ──
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  [CHECKPOINT] Removed {checkpoint_path} (run completed successfully)")

    old_checkpoint_path = args.output + ".checkpoint.json"
    if os.path.exists(old_checkpoint_path):
        os.remove(old_checkpoint_path)
        print(f"  [CHECKPOINT] Removed old {old_checkpoint_path}")


if __name__ == "__main__":
    main()