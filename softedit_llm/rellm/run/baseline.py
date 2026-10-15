"""SoftQA baseline evaluation for rankedit_llm (text analog of revlm baseline).

Runs SoftQA (QA Acc + MultiQA Acc@1 / F1 / nDCG) with no editing on the unedited
backbone, saving predictions incrementally to results/pred/ and reporting running
averages. The snapshot is the static input for the edit pipeline (find_errors).

Usage:
    python -m rellm.run.baseline --model_name qwen --dataset_name protoqa --eval_every 50
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from rellm.config_utils import configure_args
from rellm.dataset.datasets import TQADataset
from rellm.models import TQAModel
from rellm.metrics.ranked_metrics import (
    build_labels_true, soft_acc, rank_acc, set_f1, ndcg_at_k, js_distance,
)

SCORE_KEYS = ("acc", "rank_acc", "acc1", "rank_acc1", "f1", "ndcg")


def score_example(ex: dict) -> dict:
    """Per-sample SoftQA scores (QA soft/rank acc + MultiQA@1 f1/ndcg + js_dist)."""
    gold = ex.get("gold", {})
    pred_multi = ex.get("pred_multiqa", {})
    answer_counts = gold.get("answer_counts", {})
    labels_true = build_labels_true(answer_counts)

    pred_qa = ex.get("pred_qa") or ex.get("pred") or {}
    p_qa = str(pred_qa.get("label", "")).strip().lower()
    acc = soft_acc(p_qa, labels_true)
    r_acc = rank_acc(p_qa, labels_true)

    support_prob = pred_qa.get("support_prob")
    js = js_distance(answer_counts, support_prob) if (support_prob and answer_counts) else None

    pred_ranked = pred_multi.get("labels_ranked", [])
    p1 = pred_ranked[0] if pred_ranked else ""
    if labels_true and p1:
        acc1 = soft_acc(p1, labels_true)
        rank_acc1 = rank_acc(p1, labels_true)
        f1 = set_f1(pred_ranked, labels_true)
        ndcg = ndcg_at_k(pred_ranked, labels_true)
    else:
        acc1 = rank_acc1 = f1 = ndcg = 0.0

    return {
        "acc": acc, "rank_acc": r_acc,
        "acc1": acc1, "rank_acc1": rank_acc1,
        "f1": f1, "ndcg": ndcg, "js_dist": js,
    }


def aggregate_scores(examples: list) -> dict:
    if not examples:
        return {**{k: 0.0 for k in SCORE_KEYS}, "js_dist": None, "n": 0}
    sums = {k: 0.0 for k in SCORE_KEYS}
    js_sum, js_n, n = 0.0, 0, 0
    for ex in examples:
        scores = score_example(ex)
        for k in SCORE_KEYS:
            sums[k] += scores[k]
        if scores.get("js_dist") is not None:
            js_sum += scores["js_dist"]
            js_n += 1
        n += 1
    out = {k: sums[k] / n for k in SCORE_KEYS}
    out["js_dist"] = js_sum / js_n if js_n > 0 else None
    out["n"] = n
    return out


_RUNTIME_KEYS = {"idx", "scores"}


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def fmt_metrics(m: dict) -> str:
    js = m.get("js_dist")
    js_str = f"{js:.4f}" if js is not None else "\u2014"
    return (
        f"n={m['n']}  acc={m['acc']:.4f}  rank_acc={m.get('rank_acc', 0):.4f}  "
        f"acc1={m['acc1']:.4f}  rank_acc1={m.get('rank_acc1', 0):.4f}  "
        f"f1={m['f1']:.4f}  ndcg={m['ndcg']:.4f}  js={js_str}"
    )


def run_baseline(config, eval_every: int = 50):
    pred_path = os.path.join(config.pred_dir, config.fname)
    overwrite = bool(getattr(config, "overwrite", False))

    print("=" * 60, flush=True)
    print(f"SoftQA baseline | model={config.model.name} | "
          f"dataset={config.experiment.dataset_name}", flush=True)
    print(f"pred \u2192 {pred_path}", flush=True)
    print(f"eval_every={eval_every}  overwrite={overwrite}", flush=True)
    print("=" * 60, flush=True)

    seed = getattr(config, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ds = TQADataset(config)
    if config.subsample and len(ds.data) > config.subsample:
        print(f"Subsampling {len(ds.data)} \u2192 {config.subsample}", flush=True)
        ds.data = random.sample(ds.data, config.subsample)

    all_data = list(ds.data)
    done = []
    done_uids = set()

    if os.path.exists(pred_path) and not overwrite:
        done = load_json(pred_path)
        ds.reparse_snapshot(done)
        for ex in done:
            done_uids.add(str(ex["uid"]))
        print(f"Resuming: {len(done)} preds already saved", flush=True)

    remaining = [ex for ex in all_data if str(ex["uid"]) not in done_uids]
    print(f"Total={len(all_data)}  done={len(done)}  remaining={len(remaining)}", flush=True)

    if not remaining:
        final = aggregate_scores(done)
        print(f"[complete] {fmt_metrics(final)}", flush=True)
        return final

    model = TQAModel(config)
    t0 = time.time()

    for start in range(0, len(remaining), eval_every):
        chunk = remaining[start:start + eval_every]
        ds.data = chunk
        ds.set_dataloader()
        ds.task_generate(model, use_cache=False)
        for ex in ds.data:
            for k in _RUNTIME_KEYS:
                ex.pop(k, None)
        done.extend(ds.data)

        save_json(done, pred_path)
        metrics = aggregate_scores(done)
        print(f"[{metrics['n']}/{len(all_data)}] {fmt_metrics(metrics)}  "
              f"elapsed={(time.time() - t0) / 60:.1f}m", flush=True)

    final = aggregate_scores(done)
    save_json(done, pred_path)
    print("=" * 60, flush=True)
    print(f"[done] {fmt_metrics(final)}", flush=True)
    print(f"Saved preds \u2192 {pred_path}", flush=True)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoftQA baseline prediction")
    parser.add_argument("--config", type=str, default="rellm/config/config.yaml")
    parser.add_argument("--model_name", type=str, required=True, help="qwen | llama | mistral")
    parser.add_argument("--dataset_name", type=str, required=True, help="protoqa")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--subsample", type=int, default=0, help="Random subset size (0=all)")
    parser.add_argument("--eval_every", type=int, default=50, help="Save preds every N samples")
    parser.add_argument("--overwrite", action="store_true", help="Ignore cached preds and regenerate")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--label_strategy", type=str, default=None, choices=["text", "prob"],
                        help="Single-QA label parsing. Default from config.yaml.")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.editor = "baseline"
    args.task = "softqa"
    args.split = "all"
    args.suffix = ""
    config = configure_args(args, config_path=args.config)
    config.subsample = args.subsample
    config.overwrite = args.overwrite
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens

    run_baseline(config, eval_every=args.eval_every)
