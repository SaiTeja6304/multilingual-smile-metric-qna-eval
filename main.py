import argparse
import csv
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent / "pyscripts"))

from multilingual_utils import normalize_lang, get_smile_emb_model, unicode_normalize, SUPPORTED_LANGUAGES
from smile.smile import SMILE
from utils import (
    compute_rouge_score, compute_bert_score, compute_meteor_score,
    compute_exact_match, compute_sbert_score, compute_bleurt_score,
    compute_moverscore,
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multilingual SMILE Metric Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example:
                python main.py --input llm_answers.jsonl --ground-truth ground_truth.jsonl --output results.jsonl
        """,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the input JSONL file with LLM answers. "
             "Each line: {question_id, question, answer}",
    )
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to the ground-truth JSONL file. "
             "Each line: {question_id, question, answers (array), lang}",
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results.jsonl",
        help="Path to the output JSONL file (default: evaluation_results.jsonl)",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+",
        default=["smile", "rouge", "bertscore", "meteor", "exact_match", "sbert", "bleurt", "moverscore"],
        help="Which metrics to compute (default: all 8). "
             "Choices: smile, rouge, bertscore, meteor, exact_match, sbert, bleurt, moverscore",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress messages",
    )
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
    """
    Merge input (LLM answers) with ground-truth by question_id.
    
    Returns:
        list of dicts with keys: question_id, question, pred, answers, lang
    """
    # Build ground-truth lookup by question_id
    gt_lookup = {}
    for item in gt_data:
        qid = str(item.get("question_id", item.get("id", "")))
        gt_lookup[qid] = item

    merged = []
    missing_count = 0
    for item in input_data:
        qid = str(item.get("question_id", item.get("id", "")))
        gt = gt_lookup.get(qid)
        if gt is None:
            missing_count += 1
            continue

        # Normalize answers to a list
        answers = gt.get("answers", gt.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]

        lang = gt.get("lang", "en")

        merged.append({
            "question_id": qid,
            "question": item.get("question", ""),
            "pred": item.get("answer", ""),
            "answers": answers,
            "lang": lang,
        })

    if missing_count > 0:
        print(f"WARNING: {missing_count} questions in input had no matching ground-truth (skipped)")
    
    return merged


def group_by_language(merged_data: list) -> dict:
    """Group merged data by language for efficient batch processing."""
    groups = defaultdict(list)
    for item in merged_data:
        lang = normalize_lang(item["lang"])
        groups[lang].append(item)
    return dict(groups)


def compute_smile_scores(items: list, lang: str, verbose: bool = False) -> list:
    """Compute SMILE scores for a batch of items in the same language."""
    all_scores = []
    
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

    for item in items:
        pred = item["pred"]
        best_avg = 0.0
        best_hm  = 0.0

        for ref_ans in item["answers"]:
            # Set to use_ans=True mode
            qa_set = np.array([[item["question"], ref_ans, ref_ans, pred]])
            try:
                results = smile_obj.generate_scores(qa_set)
                avg_val = float(results["avg"][0])
                hm_val  = float(results["hm"][0])
                if avg_val > best_avg:
                    best_avg = avg_val
                if hm_val > best_hm:
                    best_hm = hm_val
            except Exception as e:
                if verbose:
                    print(f"  SMILE error for qid={item['question_id']}: {e}")
        best_score = {"smile_avg": best_avg, "smile_hm": best_hm}
        all_scores.append(best_score)
    
    return all_scores


def compute_metric_batch(items: list, lang: str, metric_name: str, verbose: bool = False) -> list:
    """
    Compute a single metric for a batch of items.
    For multi-reference answers, takes the max score across references.
    """
    all_scores = []

    for item in items:
        pred = item["pred"]
        best_score = -float("inf")

        for ref_ans in item["answers"]:
            ref_str = ref_ans if isinstance(ref_ans, str) else str(ref_ans)
            # Format data as expected by compute_* functions:
            # (question, answer, answer_as_syn_ans, pred)
            data_row = [(item["question"], ref_str, ref_str, pred)]

            try:
                if metric_name == "rouge":
                    result = compute_rouge_score(
                        metrics=['rougeL'], ref_data=data_row,
                        sub_metrics=['fmeasure'], ans_idx=1, lang=lang,
                    )
                    score = result['rougeL']['fmeasure'][0]

                elif metric_name == "bertscore":
                    result = compute_bert_score(inp_data=data_row, ans_idx=1, lang=lang)
                    score = result['F1'][0]

                elif metric_name == "meteor":
                    result = compute_meteor_score(inp_data=data_row, ans_idx=1, lang=lang)
                    score = result['meteor'][0]

                elif metric_name == "exact_match":
                    result = compute_exact_match(inp_data=data_row, ans_idx=1, lang=lang)
                    score = result['exact_match'][0]

                elif metric_name == "sbert":
                    result = compute_sbert_score(inp_data=data_row, ans_idx=1, lang=lang)
                    score = float(result[0])

                elif metric_name == "bleurt":
                    result = compute_bleurt_score(inp_data=data_row, ans_idx=1, lang=lang)
                    score = result['scores'][0]

                elif metric_name == "moverscore":
                    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
                    result = compute_moverscore(
                        inp_data=data_row, ans_idx=1, lang=lang, device=device,
                    )
                    score = result['scores'][0]
                else:
                    score = 0.0

            except Exception as e:
                if verbose:
                    print(f"  {metric_name} error for qid={item['question_id']}: {e}")
                score = 0.0

            if score > best_score:
                best_score = score

        all_scores.append(best_score if best_score > -float("inf") else 0.0)

    return all_scores


def run_batched_metric(items: list, lang: str, metric_name: str, verbose: bool = False) -> list:
    """
    Optimized batch computation for metrics that support it.
    """
    # Check if all items have single reference
    all_single_ref = all(len(item["answers"]) == 1 for item in items)
    
    if all_single_ref and metric_name in ("bertscore", "sbert", "rouge", "meteor", "exact_match", "bleurt", "moverscore"):
        
        data_rows = [
            (item["question"], item["answers"][0], item["answers"][0], item["pred"])
            for item in items
        ]
        
        try:
            if metric_name == "rouge":
                result = compute_rouge_score(
                    metrics=['rougeL'], ref_data=data_rows,
                    sub_metrics=['fmeasure'], ans_idx=1, lang=lang,
                )
                return result['rougeL']['fmeasure']
            
            elif metric_name == "bertscore":
                result = compute_bert_score(inp_data=data_rows, ans_idx=1, lang=lang)
                return result['F1']
            
            elif metric_name == "meteor":
                result = compute_meteor_score(inp_data=data_rows, ans_idx=1, lang=lang)
                return result['meteor']
            
            elif metric_name == "exact_match":
                result = compute_exact_match(inp_data=data_rows, ans_idx=1, lang=lang)
                return result['exact_match']
            
            elif metric_name == "sbert":
                result = compute_sbert_score(inp_data=data_rows, ans_idx=1, lang=lang)
                return result.tolist()
            
            elif metric_name == "bleurt":
                result = compute_bleurt_score(inp_data=data_rows, ans_idx=1, lang=lang)
                return result['scores']
            
            elif metric_name == "moverscore":
                device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
                result = compute_moverscore(
                    inp_data=data_rows, ans_idx=1, lang=lang, device=device,
                )
                return result['scores']
        
        except Exception as e:
            if verbose:
                print(f"  Batch {metric_name} failed, falling back to per-item: {e}")
    

    return compute_metric_batch(items, lang, metric_name, verbose)


def main():
    args = parse_arguments()

    print("Step 1: Loading input files")
    
    input_data = load_jsonl(args.input)
    gt_data = load_jsonl(args.ground_truth)
    print(f"  Input:        {len(input_data)} questions")
    print(f"  Ground-truth: {len(gt_data)} questions")


    print("\nStep 2: Merging by question_id")
    merged = merge_data(input_data, gt_data)
    print(f"  Matched: {len(merged)} questions")

    if len(merged) == 0:
        print("ERROR: No matching question_ids found. Check your files.")
        sys.exit(1)


    lang_groups = group_by_language(merged)
    print("\n  Language distribution:")
    for lang, items in sorted(lang_groups.items()):
        print(f"    {lang}: {len(items)} questions")

    # Build a mapping from question_id to index in merged for result reassembly
    qid_to_idx = {item["question_id"]: i for i, item in enumerate(merged)}

    # Initialize per-question results
    results_per_question = [{
        "question_id": item["question_id"],
        "lang": normalize_lang(item["lang"]),
    } for item in merged]

    metrics_to_run = [m.lower() for m in args.metrics]
    total_start = time.time()


    for lang, items in sorted(lang_groups.items()):
        print(f"\n{'=' * 70}")
        print(f"Processing language: {lang.upper()} ({len(items)} questions)")
        print(f"{'=' * 70}")

        # Get indices for reassembly
        indices = [qid_to_idx[item["question_id"]] for item in items]

        # SMILE
        if "smile" in metrics_to_run:
            print(f"\n  Computing SMILE for {lang}...")
            t0 = time.time()
            smile_scores = compute_smile_scores(items, lang, verbose=args.verbose)
            for idx, scores in zip(indices, smile_scores):
                results_per_question[idx].update(scores)
            print(f"    Done in {time.time() - t0:.1f}s")

        # Other metrics
        other_metrics = [m for m in metrics_to_run if m != "smile"]
        for metric_name in other_metrics:
            display_name = {
                "rouge": "ROUGE-L", "bertscore": "BERTScore", "meteor": "METEOR",
                "exact_match": "Exact Match", "sbert": "sBERT",
                "bleurt": "BLEURT", "moverscore": "MoverScore",
            }.get(metric_name, metric_name)
            
            print(f"\n  Computing {display_name} for {lang}...")
            t0 = time.time()
            scores = run_batched_metric(items, lang, metric_name, verbose=args.verbose)
            
            # Map metric name to output key
            key_map = {
                "rouge": "rouge_l", "bertscore": "bert_score_f1",
                "meteor": "meteor", "exact_match": "exact_match",
                "sbert": "sbert", "bleurt": "bleurt", "moverscore": "moverscore",
            }
            key = key_map.get(metric_name, metric_name)
            
            for idx, score in zip(indices, scores):
                results_per_question[idx][key] = float(score)
            
            print(f"    Done in {time.time() - t0:.1f}s")

    total_time = time.time() - total_start

    print("Computing aggregate summary")

    # Collect all metric keys
    metric_keys = set()
    for r in results_per_question:
        for k in r:
            if k not in ("question_id", "lang"):
                metric_keys.add(k)
    metric_keys = sorted(metric_keys)

    # Overall mean
    summary = {"question_id": "AGGREGATE_SUMMARY", "lang": "all"}
    for key in metric_keys:
        values = [r[key] for r in results_per_question if key in r]
        if values:
            summary[key] = float(np.mean(values))

    # Per-language mean
    per_lang_summary = {}
    for lang, items in sorted(lang_groups.items()):
        lang_indices = [qid_to_idx[item["question_id"]] for item in items]
        lang_summary = {"question_id": f"SUMMARY_{lang.upper()}", "lang": lang}
        for key in metric_keys:
            values = [results_per_question[i][key] for i in lang_indices if key in results_per_question[i]]
            if values:
                lang_summary[key] = float(np.mean(values))
        per_lang_summary[lang] = lang_summary

   
    print(f"\nStep 6: Writing results to {args.output}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.output.lower().endswith('.csv'):
        fieldnames = ["question_id", "lang"] + metric_keys
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Write per-question results
            for r in results_per_question:
                writer.writerow(r)
            # Write per-language summaries
            for lang_summary in per_lang_summary.values():
                writer.writerow(lang_summary)
            # Write overall summary
            writer.writerow(summary)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            # Write per-question results
            for r in results_per_question:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
            # Write per-language summaries
            for lang_summary in per_lang_summary.values():
                f.write(json.dumps(lang_summary, ensure_ascii=False) + "\n")
            
            # Write overall summary
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

   
    print(f"EVALUATION COMPLETE — {len(merged)} questions, {total_time:.1f}s total")
    
    
    print(f"\nOverall Scores (mean across all {len(merged)} questions):")
    print(f"  {'Metric':<20} {'Score':>8}")
    print(f"  {'-'*20} {'-'*8}")
    for key in metric_keys:
        if key in summary:
            print(f"  {key:<20} {summary[key]:>8.4f}")

    print(f"\nPer-Language Scores:")
    for lang, lang_summary in sorted(per_lang_summary.items()):
        n = len(lang_groups[lang])
        print(f"\n  {lang.upper()} ({n} questions):")
        for key in metric_keys:
            if key in lang_summary:
                print(f"    {key:<20} {lang_summary[key]:>8.4f}")

    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
