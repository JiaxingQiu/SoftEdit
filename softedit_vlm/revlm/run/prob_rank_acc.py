"""Offline prob_rank_acc from saved post-edit prediction dumps (no model runs).

Recomputes the soft-generality monotonic-probability metric (see
``prob_rank_acc_by_strength`` in revlm.metrics.editeval) from the per-sample
prediction snapshots a run already saved:

    pred_postedit_dir/
        reliability.json            # p_0 anchor (edit set)
        soft_generality_t0.1.json   # one file per strength
        ...
        seq/batch_000200/           # per-checkpoint dumps (sequential runs)

Usage:
    python -m revlm.run.prob_rank_acc results/pred_postedit/<editor>/<model>/<dataset> [...]

Scores the directory itself (final eval) plus every seq/batch_*/ checkpoint,
prints a table, and writes prob_rank_acc.json into each run dir.

Note: dumps written before the support_labels override landed carry a
degenerate support_prob ({label: 1.0}) on soft-gen samples; for those runs
prob_rank_acc is only meaningful for editors that attach a readout
distribution (support_prob_readout, e.g. SoftEdit).
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from revlm.metrics.editeval import chain_prob_records, prob_rank_acc_by_strength  # noqa: E402


def _load_records(path):
    with open(path) as f:
        return chain_prob_records(json.load(f))


def score_dump_dir(d):
    """prob_rank_acc table for one dump folder; None if no soft-gen dumps."""
    rel_path = os.path.join(d, "reliability.json")
    edit_recs = _load_records(rel_path) if os.path.exists(rel_path) else []
    sgen = {}
    for p in glob.glob(os.path.join(d, "soft_generality_t*.json")):
        m = re.search(r"soft_generality_t([\d.]+)\.json$", p)
        if m:
            sgen[float(m.group(1))] = _load_records(p)
    if not sgen:
        return None
    if not edit_recs:
        print(f"  [warn] {d}: no reliability.json — smallest strength has no p_0 anchor")
    return prob_rank_acc_by_strength(edit_recs, sgen)


def _fmt_row(res):
    return ", ".join(
        f"@{k}={v['prob_rank_acc']:.4f}(n={v['prob_rank_n']})"
        if v["prob_rank_acc"] is not None else f"@{k}=None(n=0)"
        for k, v in sorted(res.items())
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir", nargs="+",
                    help="pred_postedit run dir(s), e.g. results/pred_postedit/<editor>/<model>/<dataset>")
    args = ap.parse_args()

    for root in args.dump_dir:
        print(f"== {root}")
        out = {}
        final = score_dump_dir(root)
        if final:
            out["final"] = final
            print(f"  final: {_fmt_row(final)}")
        for d in sorted(glob.glob(os.path.join(root, "seq", "batch_*"))):
            res = score_dump_dir(d)
            if res:
                out.setdefault("seq", {})[os.path.basename(d)] = res
                print(f"  {os.path.basename(d)}: {_fmt_row(res)}")
        if not out:
            print("  no soft_generality dumps found")
            continue
        out_path = os.path.join(root, "prob_rank_acc.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
