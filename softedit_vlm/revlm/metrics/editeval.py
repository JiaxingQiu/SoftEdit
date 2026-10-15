from typing import Any, Dict, List, Tuple, Mapping, Sequence
import json
import os
import pandas as pd
import numpy as np
import copy
import time
import random
import gc
import torch


def move_model_device(model: Any, device: str) -> None:
	"""Move a (possibly wrapped) model to the given device if supported."""
	dev = torch.device(device) if isinstance(device, str) else device

	if hasattr(model, "device"):
		try:
			model.device = dev
		except Exception:
			pass
	if hasattr(model, "config") and hasattr(model.config, "device"):
		try:
			model.config.device = dev
		except Exception:
			pass

	if hasattr(model, "model") and hasattr(model.model, "to"):
		model.model.to(dev)
	elif hasattr(model, "to"):
		model.to(dev)


def cuda_gc() -> None:
	"""Lightweight CUDA memory cleanup."""
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def _norm(label: Any) -> str:
	return str(label).strip().lower()


def _fmt(m):
	"""Round floats in a metrics dict/value to 3 decimals for readable logs."""
	if isinstance(m, dict):
		return {k: _fmt(v) for k, v in m.items()}
	if isinstance(m, float):
		return round(m, 3)
	return m


# Aliases accepted by editeval(skip=...) -> canonical metric-set names.
_SKIP_ALIASES = {
	"reliability": "reliability", "rel": "reliability",
	"text_generality": "text_generality", "t_gen": "text_generality", "tgen": "text_generality",
	"image_generality": "image_generality", "i_gen": "image_generality", "igen": "image_generality",
	"soft_generality": "soft_generality", "soft_gen": "soft_generality", "sgen": "soft_generality",
	"blend_generality": "blend_generality", "blend_gen": "blend_generality", "bgen": "blend_generality",
	"locality": "locality", "loc": "locality",
}


def normalize_skip(skip: Sequence[str]) -> set:
	"""Canonicalize a skip list (e.g. ["soft_gen", "loc"]) to metric-set names."""
	out = set()
	for s in skip or ():
		key = str(s).strip().lower()
		if key not in _SKIP_ALIASES:
			raise ValueError(f"unknown metric set {s!r}; choose from {sorted(set(_SKIP_ALIASES.values()))}")
		out.add(_SKIP_ALIASES[key])
	return out


def generation(model: Any, edit_ds: Any) -> List[Tuple[str, str]]:
	"""Run task_generate and return (target, prediction) pairs.

	For multiqa tasks, target and prediction are the top-1 labels.
	Skips teacher-forced ``tf_acc`` — locality/hard-locality only need string
	labels, and the unrelated pool is large enough that the extra forwards
	would dominate runtime.
	"""
	edit_ds.task_generate(model, use_cache=True, compute_tf=False)
	pairs: List[Tuple[str, str]] = []
	for ex in edit_ds.data:
		gold = ex.get("gold", {})
		pred = ex.get("pred", {})
		if not pred:
			continue
		target = gold.get("label", "")
		prediction = pred.get("label", "")
		pairs.append((target, prediction))
	return pairs


def prioritized_scores(result: Dict[str, Any]) -> Dict[str, Any]:
	"""Flat dict of the metrics we track: per data group an accuracy measure and
	an alignment measure (``1 - js_dist_gate``, so higher == better). Locality
	families use ``mono`` (monotonic prob-rank) as the accuracy analog. Shared by
	the console summary and the wandb payload so both stay in sync. Values are
	None for skipped sets; callers filter as needed.
	"""
	nested = result.get("metrics") or {}
	rel = nested.get("reliability") or {}
	tgen = nested.get("text_generality") or {}
	igen = nested.get("image_generality") or {}
	soft = nested.get("soft_generality") or {}
	blend = nested.get("blend_generality") or {}
	loc = nested.get("locality") or {}

	def _acc(block):
		v = block.get("simple_acc") if isinstance(block, dict) else None
		return float(v) if v is not None else None

	def _align(block):
		v = block.get("js_dist_gate") if isinstance(block, dict) else None
		return (1.0 - float(v)) if v is not None else None

	def _mono(block):
		v = block.get("prob_rank_acc") if isinstance(block, dict) else None
		return float(v) if v is not None else None

	def _mean(vals):
		xs = [x for x in vals if x is not None]
		return sum(xs) / len(xs) if xs else None

	sgen_edit = [soft.get(f"{t}_edit") or {} for t in ("0.1", "0.3", "0.5", "0.7")]
	bgen_edit = [blend.get(f"{t}_edit") or {} for t in ("0.0", "0.3", "0.5", "0.7")]

	return {
		"rel/acc": _acc(rel), "rel/align": _align(rel),
		"igen/acc": _acc(igen), "igen/align": _align(igen),
		"tgen/acc": _acc(tgen), "tgen/align": _align(tgen),
		"loc/acc": (float(result["locality"]) if result.get("locality") is not None else _acc(loc)),
		"loc/align": _align(loc),
		"sgen_edit/acc": _mean([_acc(b) for b in sgen_edit]),
		"sgen_edit/align": _mean([_align(b) for b in sgen_edit]),
		"sloc/mono": _mono(soft.get("mono") or {}),
		"sloc/align@1.0": _align(soft.get("1.0") or {}),
		"bgen/acc": _mean([_acc(b) for b in bgen_edit]),
		"bgen/align": _mean([_align(b) for b in bgen_edit]),
		"bloc/mono": _mono(blend.get("mono") or {}),
		"bloc/align@1.0": _align(blend.get("1.0") or {}),
		"n_edits": result.get("n_edits"),
		"time_per_edit": result.get("time_per_edit"),
	}


def print_score_summary(result: Dict[str, Any]) -> None:
	"""Compact scoreboard of the prioritized metrics, in report order:
	[rel, i-gen, t-gen, loc, soft-gen, soft-loc, hard-soft-gen, hard-soft-loc].
	blend-gen/-loc are the "hard" (multi-anchor) version of soft-gen/-loc.
	"""
	s = prioritized_scores(result)

	def f(v):
		return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "

	def acc_row(name, ka, kal):
		print(f"  {name:<14} acc={f(s.get(ka))}  align={f(s.get(kal))}", flush=True)

	def loc_row(name, km, kal):
		print(f"  {name:<14} mono={f(s.get(km))}  align@1.0={f(s.get(kal))}", flush=True)

	print("=" * 50, flush=True)
	print("[scores]  acc = simple_acc | align = 1 - js_dist_gate", flush=True)
	acc_row("rel", "rel/acc", "rel/align")
	acc_row("i-gen", "igen/acc", "igen/align")
	acc_row("t-gen", "tgen/acc", "tgen/align")
	acc_row("loc", "loc/acc", "loc/align")
	acc_row("soft-gen", "sgen_edit/acc", "sgen_edit/align")
	loc_row("soft-loc", "sloc/mono", "sloc/align@1.0")
	acc_row("hard-soft-gen", "bgen/acc", "bgen/align")
	loc_row("hard-soft-loc", "bloc/mono", "bloc/align@1.0")
	print("=" * 50, flush=True)


def editeval(
		model_old: Any,
		model_new: Any,
		edit_ds: Any,
		editor: Any,
		related_texts: Mapping[int, Sequence[str]],
		related_images: Mapping[int, Sequence[Any]],
		soft_gen_images: Mapping[float, Mapping[str, Sequence[str]]] | None = None,
		blend_gen_images: Mapping[float, Mapping[str, Sequence[str]]] | None = None,
		unrelated_ds=None,
		loc_sample_size=100,
		lambda_gen: float = 1.0,
		lambda_loc: float = 1.0,
		gen_agg: str = "harmonic",
		edit_subsample_size: int = None,
		edit_time: float = None,
		dump_preds_dir: str = None,
		skip: Sequence[str] = (),
		resume_dumps: bool = False,
	) -> Dict[str, Any]:
	"""Run post-edit eval sets and return a minimal checkpoint dict.

	Returns ``{n_edits, locality, locality_hard, metrics, [edit_time,
	time_per_edit]}`` where ``metrics`` holds the full per-set metric dicts
	(reliability / t-gen / i-gen / soft-gen@strength / blend-gen@strength /
	locality). Console prints still show primary scalars; callers that want
	Acc/JS/S-Gen@k/hm should derive them from ``metrics``.

	edit_subsample_size: if set, cap edit_ds for intermediate checkpoints.
	edit_time: total editing seconds -> ``time_per_edit``.
	soft_gen_images: optional ``{strength: {uid: [path, ...]}}``.
	blend_gen_images: optional ``{strength: {uid: [path, ...]}}`` (ladder layout).
	skip: metric sets to skip for quick sessions (aliases ok, e.g.
	["soft_gen", "blend_gen", "i_gen", "loc"]); skipped sets are None in the output.
	resume_dumps: batch-only. If True, skip a metric set (or a per-strength
	soft/blend eval) when its prediction dump already exists and re-score from
	that dump. Sequential callers must leave this False so a redone peek cannot
	silently reuse a partial dump.
	"""
	skip = normalize_skip(skip)
	if skip:
		print(f"[editeval] skipping: {sorted(skip)}", flush=True)
	if resume_dumps:
		print("[editeval] resume_dumps: reusing completed prediction dumps", flush=True)

	# Intermediate peeks (edit_subsample_size set): work on a subsampled deepcopy
	# and re-apply the editor here. The caller no longer pre-applies to the full
	# edit set at peeks (that pass grew linearly with #edits), so this branch must
	# run even when the edit set is already <= edit_subsample_size.
	train_ds = edit_ds
	if edit_subsample_size is not None:
		edit_ds = copy.deepcopy(edit_ds)
		if len(edit_ds.data) > edit_subsample_size:
			rng = random.Random(333)
			edit_ds.data = rng.sample(edit_ds.data, edit_subsample_size)
		edit_ds.set_dataloader()
		_maybe_apply_ike(editor, edit_ds, train_ds)
		uids = {str(ex["uid"]) for ex in edit_ds.data}
		related_texts = {k: v for k, v in related_texts.items() if str(k) in uids}
		related_images = {k: v for k, v in related_images.items() if str(k) in uids}
		if soft_gen_images is not None:
			# soft_gen_images is nested {strength: {uid: [paths]}}
			soft_gen_images = {
				t: {k: v for k, v in per_uid.items() if str(k) in uids}
				for t, per_uid in soft_gen_images.items()
			}
		if blend_gen_images is not None:
			blend_gen_images = {
				t: {k: v for k, v in per_uid.items() if str(k) in uids}
				for t, per_uid in blend_gen_images.items()
			}

	# Post-edit prediction snapshots: only the final (global) eval dumps them;
	# `_dp(name)` returns a path when dump_preds_dir is set, else None.
	def _dp(name: str):
		return os.path.join(dump_preds_dir, f"{name}.json") if dump_preds_dir else None

	rel_metrics = None
	rel_probs: List[dict] = []  # p_0 anchor for soft-gen prob_rank_acc
	if "reliability" not in skip:
		t_rel = time.time()
		if resume_dumps:
			rel_metrics = _metrics_from_pred_dump(edit_ds, _dp("reliability"), probs_out=rel_probs)
			if rel_metrics is not None:
				print(f"[resume] reliability from dump: {time.time() - t_rel:.2f}s", flush=True)
		if rel_metrics is None:
			rel_metrics = reliability(model_new, edit_ds, label="reliability", dump_path=_dp("reliability"), probs_out=rel_probs)
			print(f"[Timing] reliability: {time.time() - t_rel:.2f}s", flush=True)
		rel = _primary_metric(rel_metrics)
		print(f"Reliability: {rel:.4f}  ({_fmt(rel_metrics)})", flush=True)

	tgen_metrics = None
	if "text_generality" not in skip:
		t_tgen = time.time()
		if resume_dumps:
			tgen_metrics = _metrics_from_pred_dump(edit_ds, _dp("text_generality"))
			if tgen_metrics is not None:
				print(f"[resume] text_generality from dump: {time.time() - t_tgen:.2f}s", flush=True)
		if tgen_metrics is None:
			tgen_metrics = text_generality(model_new, edit_ds, related_texts, editor=editor, dump_path=_dp("text_generality"))
			print(f"[Timing] text_generality: {time.time() - t_tgen:.2f}s", flush=True)
		tgen = _primary_metric(tgen_metrics)
		print(f"Text Generality: {tgen:.4f}  ({_fmt(tgen_metrics)})", flush=True)

	igen_metrics = None
	if "image_generality" not in skip:
		t_igen = time.time()
		if resume_dumps:
			igen_metrics = _metrics_from_pred_dump(edit_ds, _dp("image_generality"))
			if igen_metrics is not None:
				print(f"[resume] image_generality from dump: {time.time() - t_igen:.2f}s", flush=True)
		if igen_metrics is None:
			igen_metrics = image_generality(model_new, edit_ds, related_images, editor=editor, dump_path=_dp("image_generality"))
			print(f"[Timing] image_generality: {time.time() - t_igen:.2f}s", flush=True)
		igen = _primary_metric(igen_metrics)
		print(f"Image Generality: {igen:.4f}  ({_fmt(igen_metrics)})", flush=True)

	# Soft generality: per strength, two evals (interpolated gold + edit gold)
	# from a single forward pass.  Keys: "0.1" .. "1.0" (interp gold),
	# "0.1_edit" .. "1.0_edit" (edit gold), "mono" (prob_rank_acc).
	sgen_metrics = None
	if soft_gen_images and "soft_generality" not in skip:
		sgen_metrics = {}
		sgen_probs: Dict[float, List[dict]] = {}
		for t in sorted(soft_gen_images.keys()):
			per_uid = soft_gen_images[t]
			if not per_uid:
				continue
			t_sgen = time.time()
			m_interp = m_edit = None
			if resume_dumps:
				loaded = _softlike_from_dump(edit_ds, _dp(f"soft_generality_t{t}"), t, kind="soft")
				if loaded is not None:
					m_interp, m_edit, recs = loaded
					sgen_probs.setdefault(t, []).extend(recs)
					print(f"[resume] soft_generality@{t} from dump: {time.time() - t_sgen:.2f}s", flush=True)
			if m_interp is None:
				m_interp, m_edit = soft_generality(
					model_new, edit_ds, per_uid, t=t, editor=editor,
					dump_path=_dp(f"soft_generality_t{t}"),
					probs_out=sgen_probs.setdefault(t, []))
				print(f"[Timing] soft_generality@{t}: {time.time() - t_sgen:.2f}s", flush=True)
			sgen_metrics[str(t)] = m_interp
			sgen_metrics[f"{t}_edit"] = m_edit
			p = _primary_metric(m_interp) if m_interp else 0.0
			print(f"Soft Generality@{t}: {p:.4f}  ({_fmt(m_interp)})", flush=True)
			p_edit = _primary_metric(m_edit) if m_edit else 0.0
			print(f"Soft Gen (vs edit)@{t}: {p_edit:.4f}  ({_fmt(m_edit)})", flush=True)
		# Cross-strength monotonicity of p(target_label)
		pra = prob_rank_acc_by_strength(rel_probs, sgen_probs)
		for k, v in pra.items():
			if isinstance(sgen_metrics.get(k), dict):
				sgen_metrics[k].update(v)
		# Aggregate monotonicity into a single "mono" key
		all_pra = [v for k, v in sorted(pra.items()) if v.get("prob_rank_acc") is not None]
		if all_pra:
			mono_acc = sum(v["prob_rank_acc"] for v in all_pra) / len(all_pra)
			mono_n = sum(v["prob_rank_n"] for v in all_pra)
			sgen_metrics["mono"] = {"prob_rank_acc": mono_acc, "prob_rank_n": mono_n}
		else:
			sgen_metrics["mono"] = {"prob_rank_acc": None, "prob_rank_n": 0}
		print("Soft Gen prob_rank_acc: "
		      + ", ".join(f"@{k}={v['prob_rank_acc']:.4f} (n={v['prob_rank_n']})"
		                  if v["prob_rank_acc"] is not None else f"@{k}=None (n=0)"
		                  for k, v in sorted(pra.items())), flush=True)
		print(f"Soft Gen mono: {_fmt(sgen_metrics['mono'])}", flush=True)

	# Blend generality: per strength, same two-gold eval as soft_gen.
	# Keys: "0.0" .. "1.0" (interp gold), "0.0_edit" .. "1.0_edit" (edit gold),
	# "mono" (prob_rank_acc across strengths over all label anchors -> b-loc).
	bgen_metrics = None
	if blend_gen_images and "blend_generality" not in skip:
		bgen_metrics = {}
		bgen_probs: Dict[float, List[dict]] = {}
		for t in sorted(blend_gen_images.keys()):
			per_uid = blend_gen_images[t]
			if not per_uid:
				continue
			t_bgen = time.time()
			m_interp = m_edit = None
			if resume_dumps:
				loaded = _softlike_from_dump(edit_ds, _dp(f"blend_generality_t{t}"), t, kind="blend")
				if loaded is not None:
					m_interp, m_edit, recs = loaded
					bgen_probs.setdefault(t, []).extend(recs)
					print(f"[resume] blend_generality@{t} from dump: {time.time() - t_bgen:.2f}s", flush=True)
			if m_interp is None:
				m_interp, m_edit = blend_generality(
					model_new, edit_ds, per_uid, t=t, editor=editor,
					dump_path=_dp(f"blend_generality_t{t}"),
					probs_out=bgen_probs.setdefault(t, []))
				print(f"[Timing] blend_generality@{t}: {time.time() - t_bgen:.2f}s", flush=True)
			bgen_metrics[str(t)] = m_interp
			bgen_metrics[f"{t}_edit"] = m_edit
			p = _primary_metric(m_interp) if m_interp else 0.0
			print(f"Blend Generality@{t}: {p:.4f}  ({_fmt(m_interp)})", flush=True)
			p_edit = _primary_metric(m_edit) if m_edit else 0.0
			print(f"Blend Gen (vs edit)@{t}: {p_edit:.4f}  ({_fmt(m_edit)})", flush=True)
		# Cross-strength monotonicity of p(anchor label) -> b-loc "mono".
		# All label anchors; reliability p_0 (original image) anchors the smallest
		# strength, same machinery as soft-gen mono.
		pra = prob_rank_acc_by_strength(rel_probs, bgen_probs)
		for k, v in pra.items():
			if isinstance(bgen_metrics.get(k), dict):
				bgen_metrics[k].update(v)
		all_pra = [v for k, v in sorted(pra.items()) if v.get("prob_rank_acc") is not None]
		if all_pra:
			mono_acc = sum(v["prob_rank_acc"] for v in all_pra) / len(all_pra)
			mono_n = sum(v["prob_rank_n"] for v in all_pra)
			bgen_metrics["mono"] = {"prob_rank_acc": mono_acc, "prob_rank_n": mono_n}
		else:
			bgen_metrics["mono"] = {"prob_rank_acc": None, "prob_rank_n": 0}
		print(f"Blend Gen mono: {_fmt(bgen_metrics['mono'])}", flush=True)

	loc_metrics = None
	if "locality" not in skip:
		t_loc = time.time()
		if resume_dumps:
			loc_metrics = _locality_from_dump(_dp("locality"))
			if loc_metrics is not None:
				print(f"[resume] locality from dump: {time.time() - t_loc:.2f}s", flush=True)
		if loc_metrics is None:
			loc_metrics = locality(
				model_old,
				model_new,
				train_ds,
				unrelated_ds=unrelated_ds,
				sample_size=loc_sample_size,
				editor=editor,
				dump_path=_dp("locality"),
			)
			print(f"[Timing] locality: {time.time() - t_loc:.2f}s", flush=True)
		print(f"Locality: {loc_metrics['simple_acc']:.4f}  ({_fmt(loc_metrics)})", flush=True)

	# Saved checkpoint = nested full metric dicts + locality + run metadata.
	# Top-level Acc/JS/S-Gen@k/hm scalars are intentionally omitted; derive them
	# later from `metrics` (wandb / ee_summary already do this).
	n_edits = len(edit_ds.data)
	result = {
		"n_edits": float(n_edits),
		"locality": float(loc_metrics["simple_acc"]) if loc_metrics else None,
		"locality_hard": float(loc_metrics["hard_acc"]) if loc_metrics else None,
		"metrics": {
			"reliability": rel_metrics,
			"text_generality": tgen_metrics,
			"image_generality": igen_metrics,
			"soft_generality": sgen_metrics,
			"blend_generality": bgen_metrics,
			"locality": loc_metrics,
		},
	}
	if edit_time is not None and n_edits > 0:
		result["edit_time"] = float(edit_time)
		result["time_per_edit"] = float(edit_time / n_edits)
	print_score_summary(result)
	return result


def _primary_metric(metrics: Any) -> float:
    """Collapse a task-metric dict (or float) into a single scalar for hm/printing.

    postedit_qa / qa -> soft accuracy ("acc")
    softqa           -> single-QA soft accuracy ("qa"/"acc")
    multiqa          -> top-1 soft accuracy ("acc_at_1")
    """
    if isinstance(metrics, dict):
        if "acc" in metrics:
            return float(metrics["acc"])
        if "qa" in metrics and isinstance(metrics["qa"], dict):
            return float(metrics["qa"].get("acc", 0.0))
        if "acc_at_1" in metrics:
            return float(metrics["acc_at_1"])
        return 0.0
    return float(metrics)


def reliability(model_new: Any, edit_ds: Any, label: str = None, dump_path: str = None,
                probs_out: List[dict] = None):
    """Task-metric dict for the edited model on `edit_ds` (side-effect free).

    Returns whatever the dataset's task engineer reports:
      qa     -> {"acc", "rank_acc", "n"}
      softqa -> {"qa": {...}, "multiqa": {...}}   (full 6 metrics)
      multiqa-> {"acc_at_1", "rank_acc_at_1", "f1", "ndcg_at_k", "n"}

    When config.debug is set and `label` is given, prints per-sample
    gold/pred diagnostics for this eval set (off by default).

    If `dump_path` is given, the per-sample post-edit predictions (gold + pred)
    are snapshotted to that JSON file -- a near-free by-product of the eval that
    lets any metric be recomputed offline without re-running the model.

    If `probs_out` (a list) is given, per-sample probability records for
    prob_rank_acc are appended to it (see ``chain_prob_records``).
    """
    ds = copy.deepcopy(edit_ds)
    ds.task_generate(model_new, use_cache=True)
    if label and getattr(edit_ds.config, "debug", False):
        from revlm.run.edit_utils import debug_samples  # lazy: avoid circular import
        debug_samples(ds, label)
    metrics = ds.task_engineer.eval(ds)
    if probs_out is not None:
        probs_out.extend(chain_prob_records(ds.data))
    if dump_path:
        try:
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            ds.snap(out_path=dump_path)
        except Exception as e:
            print(f"[dump] post-edit prediction snapshot failed for {dump_path}: {e}", flush=True)
    return metrics


def chain_prob_records(data: List[dict]) -> List[dict]:
    """Per-sample probability records for prob_rank_acc, from live ds.data or a
    loaded prediction dump. Prefers a retrieval editor's readout distribution
    over the model's own support_prob (same convention as js_dist)."""
    recs = []
    for ex in data:
        pred = ex.get("pred") or ex.get("pred_qa") or {}
        sp = pred.get("support_prob_readout") or pred.get("support_prob") or {}
        recs.append({
            "uid": str(ex.get("uid")),
            "gold_label": _norm((ex.get("gold") or {}).get("label", "")),
            "probs": {_norm(k): float(v) for k, v in sp.items()},
        })
    return recs


def _load_pred_dump(path: str):
    """Return a prediction-dump list, or None if missing/truncated/empty."""
    if not path or not os.path.isfile(path) or os.path.getsize(path) <= 2:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    return data


def _eval_pred_dump(edit_ds: Any, data: List[dict], probs_out: List[dict] = None):
    """Recompute task metrics from a saved prediction dump (no model forward)."""
    ds = copy.deepcopy(edit_ds)
    ds.data = data
    if getattr(ds, "task_engineer", None) is None:
        ds.set_dataloader()
    else:
        for i, ex in enumerate(ds.data):
            ex["idx"] = i
            ds.task_engineer.eng_golds(ex)
    metrics = ds.task_engineer.eval(ds)
    if probs_out is not None:
        probs_out.extend(chain_prob_records(ds.data))
    return metrics


def _metrics_from_pred_dump(edit_ds: Any, dump_path: str, probs_out: List[dict] = None):
    data = _load_pred_dump(dump_path)
    if data is None:
        return None
    try:
        return _eval_pred_dump(edit_ds, data, probs_out=probs_out)
    except Exception as e:
        print(f"[resume] failed to score dump {dump_path}: {e}", flush=True)
        return None


def _softlike_from_dump(edit_ds: Any, dump_path: str, t: float, kind: str):
    """Rebuild (metrics_interp, metrics_edit, prob_records) from a soft/blend dump.

    Dumps are written after the interpolated gold is swapped in, so interp metrics
    are scored as-is; edit-gold metrics restore ``answer_counts`` from ``edit_ds``.
    """
    from pathlib import Path as _Path
    from revlm.metrics.utils.blend_gen import blend_gen_label_from_path

    data = _load_pred_dump(dump_path)
    if data is None:
        return None

    def _counts(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return {}
        return x if isinstance(x, dict) else {}

    uid_to_ec = {}
    for ex in edit_ds.data:
        uid_to_ec[str(ex.get("uid"))] = _counts(
            (ex.get("gold") or {}).get("answer_counts") or ex.get("answer_counts") or {}
        )

    try:
        ds = copy.deepcopy(edit_ds)
        ds.data = copy.deepcopy(data)
        if getattr(ds, "task_engineer", None) is None:
            ds.set_dataloader()
        else:
            for i, ex in enumerate(ds.data):
                ex["idx"] = i
                ds.task_engineer.eng_golds(ex)
        metrics_interp = ds.task_engineer.eval(ds)
        recs = chain_prob_records(ds.data)

        edit_ac = []
        for ex in ds.data:
            img = ex.get("image") or ex.get("image_path") or ""
            lab = (blend_gen_label_from_path(img) if kind == "blend"
                   else _Path(str(img)).stem.rsplit("_t", 1)[0])
            ec = dict(uid_to_ec.get(str(ex.get("uid")), {}))
            if lab not in ec:
                ec[lab] = 0
            total = sum(ec.values()) or 1
            edit_ac.append({l: c / total for l, c in ec.items()})
        for ex, eac in zip(ds.data, edit_ac):
            ex["answer_counts"] = eac
            ex["labels_ranked"] = [l for l, _ in sorted(eac.items(), key=lambda x: -x[1]) if _ > 0]
            ds.task_engineer.eng_golds(ex)
        metrics_edit = ds.task_engineer.eval(ds)
    except Exception as e:
        print(f"[resume] failed to score dump {dump_path}: {e}", flush=True)
        return None
    return metrics_interp, metrics_edit, recs


def _locality_from_dump(dump_path: str):
    """Recompute locality metrics from a dump that carries ``loc_gold`` + ``pred``."""
    from revlm.metrics.ranked_metrics import (
        build_labels_true, js_distance, tau_readout, ndcg_readout,
    )
    data = _load_pred_dump(dump_path)
    if data is None:
        return None
    if not any(isinstance(ex.get("loc_gold"), dict) for ex in data):
        return None

    def _open_readout(pred):
        spo = (pred or {}).get("support_prob_open") or {}
        return {_norm(k): float(v) for k, v in spo.items()
                if str(k) != "__OTHER__" and v is not None}

    n = 0
    acc_sum = 0.0
    js_sum = 0.0
    js_n = 0
    tau_sum = 0.0
    ndcg_sum = 0.0
    rank_n = 0
    try:
        for ex in data:
            pred_old = ex.get("loc_gold") or {}
            pred_new = ex.get("pred") or {}
            if not pred_old or not pred_new:
                continue
            n += 1
            old_label = _norm(pred_old.get("label", ""))
            new_label = _norm(pred_new.get("label", ""))
            acc_sum += float(bool(old_label) and old_label == new_label)
            old_spo = _open_readout(pred_old)
            new_spo = _open_readout(pred_new)
            if old_spo and new_spo:
                js_sum += js_distance(old_spo, new_spo)
                js_n += 1
                gold_tiers = build_labels_true(old_spo)
                tau_sum += tau_readout(new_spo, gold_tiers, label2key=None)
                ndcg_sum += ndcg_readout(new_spo, gold_tiers, label2key=None)
                rank_n += 1
    except Exception as e:
        print(f"[resume] failed to score locality dump {dump_path}: {e}", flush=True)
        return None
    if n == 0:
        return None
    acc = acc_sum / n
    js = js_sum / js_n if js_n > 0 else None
    tau = tau_sum / rank_n if rank_n > 0 else None
    ndcg = ndcg_sum / rank_n if rank_n > 0 else None
    return {
        "simple_acc": acc,
        "hard_acc": acc,
        "rank_acc": acc,
        "acc": acc,
        "js_dist": js,
        "js_dist_gate": js,
        "tau_readout": tau,
        "tau_readout_gate": tau,
        "ndcg_readout": ndcg,
        "ndcg_readout_gate": ndcg,
        "n": n,
    }


def prob_rank_acc_by_strength(
    edit_recs: List[dict],
    sgen_recs_by_strength: Dict[float, List[dict]],
) -> Dict[str, Dict[str, Any]]:
    """Monotonic-probability accuracy for soft generality.

    Each soft-gen image interpolates the original edit image toward one target
    label with strength t; a chain is keyed by (uid, target_label). Per
    strength, a chain scores 1 if the model's probability of the target label
    strictly exceeds its probability at the previous strength (the original
    edit sample anchors the smallest strength, so its records come from the
    reliability pass). Ties and missing probs skip the pair; ``prob_rank_n``
    reports how many pairs were scored.

    Returns {strength_str: {"prob_rank_acc": float|None, "prob_rank_n": int}}.
    """
    p0 = {r["uid"]: r["probs"] for r in edit_recs}
    strengths = sorted(sgen_recs_by_strength)
    pt: Dict[float, Dict[tuple, float]] = {}
    for t in strengths:
        pt[t] = {}
        for r in sgen_recs_by_strength[t]:
            p = r["probs"].get(r["gold_label"])
            if p is not None:
                pt[t][(r["uid"], r["gold_label"])] = p
    out = {}
    for i, t in enumerate(strengths):
        prev = pt[strengths[i - 1]] if i > 0 else {
            key: p0[key[0]][key[1]]
            for key in pt[t] if key[1] in p0.get(key[0], {})
        }
        wins = n = 0
        for key, p in pt[t].items():
            pp = prev.get(key)
            if pp is None or p == pp:
                continue
            n += 1
            wins += int(p > pp)
        out[str(t)] = {"prob_rank_acc": wins / n if n else None, "prob_rank_n": n}
    return out


def locality(
    model_old: Any,
    model_new: Any,
    edit_ds: Any,
    unrelated_ds=None,
    sample_size=None,
    editor: Any = None,
    dump_path: str = None,
) -> Dict[str, float]:
    """Behaviour drift on unrelated inputs, scored with MODEL_OLD as the gold.

    When ``unrelated_ds`` is provided (the baseline-correct locality set built
    once by ``find_errors``), ``sample_size`` items are drawn from it.  Otherwise
    falls back to full parquet minus *all* edit uids on ``edit_ds``.
    """
    from revlm.metrics.ranked_metrics import (
        build_labels_true, js_distance, tau_readout, ndcg_readout,
    )

    if unrelated_ds is None:
        unrelated_ds = copy.deepcopy(edit_ds)
        full_df = edit_ds.load_df()
        used_uids = {str(ex.get("uid")) for ex in edit_ds.data}
        mask = ~full_df["uid"].astype(str).isin(used_uids)
        pool_df = full_df.loc[mask].reset_index(drop=True)
        if pool_df.empty:
            raise ValueError("No unrelated inputs found")
        if sample_size is not None:
            pool_df = pool_df.sample(
                n=min(sample_size, len(pool_df)),
                random_state=getattr(edit_ds.config, "seed", 333),
            )
        unrelated_ds.data = unrelated_ds.df2data(pool_df)
        unrelated_ds.set_dataloader()
    elif sample_size is not None and len(unrelated_ds.data) > sample_size:
        unrelated_ds = copy.deepcopy(unrelated_ds)
        base_seed = getattr(edit_ds.config, "seed", 333)
        rng = random.Random(base_seed + len(edit_ds.data))
        unrelated_ds.data = rng.sample(unrelated_ds.data, sample_size)
        unrelated_ds.set_dataloader()

    ds_old = copy.deepcopy(unrelated_ds)
    ds_new = copy.deepcopy(unrelated_ds)
    # Only first-token readouts are needed: skip the continuation gate (the
    # *_gate columns mirror the plain ones) and teacher forcing for efficiency.
    for d in (ds_old, ds_new):
        if getattr(d, "config", None) is not None:
            try:
                d.config.continuation_gate = False
            except Exception:
                pass

    target_device = getattr(edit_ds.config, "device", "cuda")
    if isinstance(target_device, str):
        target_device = torch.device(target_device)

    # Base-model pass (gold). Scored generation records first-token readouts.
    move_model_device(model_new, "cpu")
    cuda_gc()
    move_model_device(model_old, target_device)
    ds_old.set_dataloader()
    ds_old.task_generate(model_old, use_cache=True, compute_tf=False)
    move_model_device(model_old, "cpu")
    cuda_gc()

    # Edited-model pass (prediction). Retrieval editors are applied AFTER
    # set_dataloader (eng_prompt re-runs inside it and would wipe an injected
    # fact) and after model_new is back on the GPU (SoftEdit's readout forwards
    # the model). A false-positive application here is exactly what we measure.
    ds_new.set_dataloader()
    move_model_device(model_new, target_device)
    _maybe_apply_ike(editor, ds_new, edit_ds)
    ds_new.task_generate(model_new, use_cache=True, compute_tf=False)
    move_model_device(model_new, "cpu")
    cuda_gc()

    def _open_readout(pred):
        spo = pred.get("support_prob_open") or {}
        return {_norm(k): float(v) for k, v in spo.items()
                if str(k) != "__OTHER__" and v is not None}

    n = 0
    acc_sum = 0.0
    js_sum = 0.0
    js_n = 0
    tau_sum = 0.0
    ndcg_sum = 0.0
    rank_n = 0
    for ex_old, ex_new in zip(ds_old.data, ds_new.data):
        pred_old = ex_old.get("pred") or {}
        pred_new = ex_new.get("pred") or {}
        if not pred_old or not pred_new:
            continue
        n += 1
        old_label = _norm(pred_old.get("label", ""))
        new_label = _norm(pred_new.get("label", ""))
        acc_sum += float(bool(old_label) and old_label == new_label)

        old_spo = _open_readout(pred_old)
        new_spo = _open_readout(pred_new)
        if old_spo and new_spo:
            js_sum += js_distance(old_spo, new_spo)
            js_n += 1
            gold_tiers = build_labels_true(old_spo)
            tau_sum += tau_readout(new_spo, gold_tiers, label2key=None)
            ndcg_sum += ndcg_readout(new_spo, gold_tiers, label2key=None)
            rank_n += 1

        # Stash the base-model reference on the edited example so the dump carries
        # gold (old) + pred (new) for offline rescoring.
        ex_new["loc_gold"] = {
            "label": pred_old.get("label", ""),
            "answer": pred_old.get("answer", ""),
            "support_prob_open": pred_old.get("support_prob_open"),
            "open_label2key": pred_old.get("open_label2key"),
        }

    move_model_device(model_new, target_device)

    if n == 0:
        return {"simple_acc": 0.0, "hard_acc": 0.0, "rank_acc": 0.0, "acc": 0.0,
                "js_dist": None, "js_dist_gate": None,
                "tau_readout": None, "tau_readout_gate": None,
                "ndcg_readout": None, "ndcg_readout_gate": None, "n": 0}

    acc = acc_sum / n
    js = js_sum / js_n if js_n > 0 else None
    tau = tau_sum / rank_n if rank_n > 0 else None
    ndcg = ndcg_sum / rank_n if rank_n > 0 else None

    # Dump per-sample base reference + edited preds, same mechanism as every
    # other eval set (makedirs + snap, guarded so a dump failure never aborts
    # the run).
    if dump_path:
        try:
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            ds_new.snap(out_path=dump_path)
        except Exception as e:
            print(f"[dump] locality prediction snapshot failed for {dump_path}: {e}", flush=True)

    return {
        "simple_acc": acc,
        "hard_acc": acc,
        "rank_acc": acc,
        "acc": acc,
        "js_dist": js,
        "js_dist_gate": js,
        "tau_readout": tau,
        "tau_readout_gate": tau,
        "ndcg_readout": ndcg,
        "ndcg_readout_gate": ndcg,
        "n": n,
    }


def _debug_locality_samples(data, targets, preds_new, k=5):
	"""Show whether the editor actuated on (unrelated) locality samples.

	For retrieval editors (SoftEdit/ReasonEdit/IKE/GT) `apply_to_dataset` ran on
	this dataset, so each example carries the evidence of a (false-positive)
	application: a fact prepended to the prompt (prompt mode), an armed codebook
	entry `_se_entry` (hidden_state mode), or an `answer_override` (logit mode).
	Prints aggregate counts plus the first `k` samples — prioritizing FLIPs, the
	ones that cost locality — with the retrieval diagnostics (d1, entry uid).
	Weight editors carry none of these markers and just get the flip summary.
	"""
	rows = []
	n_fact = n_arm = n_ovr = n_flip = 0
	for ex, t, p in zip(data, targets, preds_new):
		prompt = ex.get("prompt") or ""
		fact = bool(ex.get("prompt_orig")) and prompt != ex.get("prompt_orig")
		armed = ex.get("_se_entry") is not None
		ovr = bool(ex.get("answer_override"))
		flip = str(t).strip().lower() != str(p).strip().lower()
		n_fact += fact; n_arm += armed; n_ovr += ovr; n_flip += flip
		rows.append((ex, t, p, fact, armed, ovr, flip))

	n = len(rows)
	print(f"[locality dbg] n={n} flips={n_flip} | applied: fact={n_fact} "
	      f"v0_armed={n_arm} override={n_ovr}", flush=True)
	# FLIPs first (they explain the score), then applied-but-survived samples.
	rows.sort(key=lambda r: (not r[6], not (r[3] or r[4] or r[5])))
	for ex, t, p, fact, armed, ovr, flip in rows[:k]:
		dbg = ex.get("readout_dbg") or {}
		marks = "".join(m for m, on in (("FACT ", fact), ("V0 ", armed),
		                                ("OVR ", ovr)) if on) or "raw "
		d1 = dbg.get("d1")
		retr = (f"d1={d1:.2f} entry={dbg.get('entry_uid')}" if d1 is not None
		        else "no-retrieval")
		print(f"[locality dbg] {'FLIP' if flip else ' ok '} uid={ex.get('uid')} "
		      f"{marks}| {retr} | old '{t}' -> new '{p}'", flush=True)
		print(f"[locality dbg]      prompt: {prompt_head(ex)}", flush=True)


def prompt_head(ex, n=200):
	"""First n chars of the applied prompt (shows the injected fact verbatim)."""
	return str(ex.get("prompt", ""))[:n].replace("\n", " ")


def _filter_correct(pairs: List[Tuple[str, str]]) -> Tuple[List[int], List[str]]:
	"""Filter to indices where prediction matches target. Returns (indices, targets)."""
	indices = [i for i, (t, p) in enumerate(pairs) if str(t).strip().lower() == str(p).strip().lower()]
	targets = [pairs[i][0] for i in indices]
	return indices, targets


def _maybe_apply_ike(
    editor: Any,
    edit_ds: Any,
    train_ds: Any,
) -> None:
    if editor is None:
        return
    cfg = getattr(train_ds, "config", None)
    editor_cfg = getattr(cfg, "editor", None)
    editor_name = getattr(editor_cfg, "_name", None) if editor_cfg is not None else None
    # `_soft` retrieval variants dispatch like their base editor.
    if editor_name and editor_name.endswith("_soft"):
        editor_name = editor_name[:-5]
    # softedit_task (and *_* variants) share the retrieval apply path
    if editor_name in ["ike", "reasonedit", "gt", "softedit"] or (
            editor_name and editor_name.startswith("softedit_")):
        if hasattr(editor, "model") and hasattr(editor, "wrapper"):
            if hasattr(editor.model, "eval"):
                editor.model.eval()
        if hasattr(editor, "apply_to_dataset"):
            cfg = getattr(train_ds, "config", None)
            if cfg and hasattr(editor, "plot_k_dist"):
                editor.plot_k_dist = getattr(cfg, "plot_k_dist", False)
            editor.apply_to_dataset(edit_ds)
            if cfg and hasattr(editor, "plot_k_dist"):
                editor.plot_k_dist = False


def text_generality(
    model_new: Any,
    edit_ds: Any,
    related_texts: Dict[str, List[str]],
    editor: Any = None,
    dump_path: str = None,
) -> float:
    """Accuracy on paraphrased/related texts using the same images."""
    df = edit_ds.load_df()
    ds = copy.deepcopy(edit_ds)
    related_df = pd.DataFrame(
        (
            (uid, question_variant)
            for uid, variants in related_texts.items()
            for question_variant in variants
        ),
        columns=["uid", "question"],
    )
    related_df = related_df.merge(
        df.drop(columns=["question"]),
        on="uid",
        how="left",
    )
    ds.data = ds.df2data(related_df)
    ds.set_dataloader()

    _maybe_apply_ike(editor, ds, edit_ds)

    return reliability(model_new, ds, label="text_generality", dump_path=dump_path)

def image_generality(
    model_new: Any,
    edit_ds: Any,
    related_images: Dict[str, List[str]],
    editor: Any = None,
    dump_path: str = None,
) -> float:
    """Accuracy on related images using the same questions."""
    df = edit_ds.load_df()
    ds = copy.deepcopy(edit_ds)
    related_df = pd.DataFrame(
        (
            (uid, image_path_variant)
            for uid, image_paths in related_images.items()
            for image_path_variant in image_paths
        ),
        columns=["uid", "image_path"],
    )
    related_df = related_df.merge(
        df.drop(columns=["image_path"]),
        on="uid",
        how="left",
    )
    ds.data = ds.df2data(related_df) 
    ds.set_dataloader()

    _maybe_apply_ike(editor, ds, edit_ds)

    return reliability(model_new, ds, label="image_generality", dump_path=dump_path)


def soft_generality(
    model_new: Any,
    edit_ds: Any,
    soft_gen_images: Dict[str, List[str]],
    t: float = 1.0,
    editor: Any = None,
    dump_path: str = None,
    probs_out: List[dict] = None,
) -> Dict[str, Any]:
    """Metrics on soft-generality images, evaluated against two golds.

    Returns ``(metrics_interp, metrics_edit)`` where each is the full
    task-metric dict.  Both share a single forward pass.

    *   ``metrics_interp`` — gold = ``(1-t)*p_edit + t*one_hot(target)``
    *   ``metrics_edit``   — gold = edit's original ``answer_counts``

    support_prob is scored over the edit sample's ORIGINAL label set
    (via ex["support_labels"]) so it is comparable across strengths and with
    the edit sample itself (prob_rank_acc).
    """
    from pathlib import Path

    df = edit_ds.load_df()
    ds = copy.deepcopy(edit_ds)

    rows = []
    for uid, image_paths in soft_gen_images.items():
        for img_path in image_paths:
            stem = Path(img_path).stem  # e.g. "cat_t1.0"
            label = stem.rsplit("_t", 1)[0]
            rows.append({"uid": uid, "image_path": img_path, "_soft_gen_label": label})

    if not rows:
        return {}, {}

    related_df = pd.DataFrame(rows)
    related_df = related_df.merge(
        df.drop(columns=["image_path"]),
        on="uid",
        how="left",
    )

    def _counts(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return {}
        return x if isinstance(x, dict) else {}

    # Capture per-row: support set, edit counts, target label
    support = []
    edit_counts_list = []
    targets = []
    for ac, lab in zip(related_df["answer_counts"], related_df["_soft_gen_label"]):
        ec = _counts(ac)
        labels = list(ec)
        if lab not in labels:
            labels.append(lab)
        support.append(labels)
        edit_counts_list.append(ec)
        targets.append(lab)

    # -- Build dataset with EDIT distribution as gold (for forward pass) --
    # The edit distribution includes the target label (count 0 if absent).
    edit_ac = []
    for ec, lab in zip(edit_counts_list, targets):
        d = dict(ec)
        if lab not in d:
            d[lab] = 0
        total = sum(d.values()) or 1
        edit_ac.append({l: c / total for l, c in d.items()})

    related_df["answer_counts"] = [json.dumps(d) for d in edit_ac]
    related_df = related_df.drop(columns=["_soft_gen_label"])

    ds.data = ds.df2data(related_df)
    for ex, labels in zip(ds.data, support):
        ex["support_labels"] = labels
    ds.set_dataloader()

    _maybe_apply_ike(editor, ds, edit_ds)

    # --- Single forward pass ---
    ds_gen = copy.deepcopy(ds)
    ds_gen.task_generate(model_new, use_cache=True)

    # --- Eval 1: edit distribution as gold ---
    metrics_edit = ds_gen.task_engineer.eval(ds_gen)
    if probs_out is not None:
        probs_out.extend(chain_prob_records(ds_gen.data))

    # --- Eval 2: interpolated gold (swap answer_counts + rebuild golds) ---
    interp_ac = []
    for ec, lab, t_val in zip(edit_ac, targets, [t] * len(targets)):
        interp = {l: (1 - t_val) * p + (t_val if l == lab else 0.0)
                  for l, p in ec.items()}
        interp_ac.append(interp)

    for ex, iac in zip(ds_gen.data, interp_ac):
        ex["answer_counts"] = iac
        ex["labels_ranked"] = [l for l, _ in sorted(iac.items(), key=lambda x: -x[1]) if _ > 0]
        ds_gen.task_engineer.eng_golds(ex)

    metrics_interp = ds_gen.task_engineer.eval(ds_gen)

    if dump_path:
        try:
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            ds_gen.snap(out_path=dump_path)
        except Exception as e:
            print(f"[dump] post-edit prediction snapshot failed for {dump_path}: {e}", flush=True)

    return metrics_interp, metrics_edit


def blend_generality(
    model_new: Any,
    edit_ds: Any,
    blend_gen_images: Dict[str, List[str]],
    t: float = 0.0,
    editor: Any = None,
    dump_path: str = None,
    probs_out: List[dict] = None,
) -> Dict[str, Any]:
    """Metrics on blend-generality ladder images, evaluated against two golds.

    Returns ``(metrics_interp, metrics_edit)`` — same structure as
    ``soft_generality``.

    *   ``metrics_interp`` — gold = ``(1-t)*p_edit + t*one_hot(target)``
    *   ``metrics_edit``   — gold = edit's original ``answer_counts``

    Label is parsed from the directory name (``tune/{label}/t{t}.png``).
    """
    from revlm.metrics.utils.blend_gen import blend_gen_label_from_path

    df = edit_ds.load_df()
    ds = copy.deepcopy(edit_ds)

    rows = []
    for uid, image_paths in blend_gen_images.items():
        for img_path in image_paths:
            label = blend_gen_label_from_path(img_path)
            rows.append({"uid": uid, "image_path": img_path, "_blend_gen_label": label})

    if not rows:
        return {}, {}

    related_df = pd.DataFrame(rows)
    related_df = related_df.merge(
        df.drop(columns=["image_path"]),
        on="uid",
        how="left",
    )

    def _counts(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return {}
        return x if isinstance(x, dict) else {}

    support = []
    edit_counts_list = []
    targets = []
    for ac, lab in zip(related_df["answer_counts"], related_df["_blend_gen_label"]):
        ec = _counts(ac)
        labels = list(ec)
        if lab not in labels:
            labels.append(lab)
        support.append(labels)
        edit_counts_list.append(ec)
        targets.append(lab)

    edit_ac = []
    for ec, lab in zip(edit_counts_list, targets):
        d = dict(ec)
        if lab not in d:
            d[lab] = 0
        total = sum(d.values()) or 1
        edit_ac.append({l: c / total for l, c in d.items()})

    related_df["answer_counts"] = [json.dumps(d) for d in edit_ac]
    related_df = related_df.drop(columns=["_blend_gen_label"])

    ds.data = ds.df2data(related_df)
    for ex, labels in zip(ds.data, support):
        ex["support_labels"] = labels
    ds.set_dataloader()

    _maybe_apply_ike(editor, ds, edit_ds)

    ds_gen = copy.deepcopy(ds)
    ds_gen.task_generate(model_new, use_cache=True)

    metrics_edit = ds_gen.task_engineer.eval(ds_gen)
    if probs_out is not None:
        probs_out.extend(chain_prob_records(ds_gen.data))

    interp_ac = []
    for ec, lab, t_val in zip(edit_ac, targets, [t] * len(targets)):
        interp = {l: (1 - t_val) * p + (t_val if l == lab else 0.0)
                  for l, p in ec.items()}
        interp_ac.append(interp)

    for ex, iac in zip(ds_gen.data, interp_ac):
        ex["answer_counts"] = iac
        ex["labels_ranked"] = [l for l, _ in sorted(iac.items(), key=lambda x: -x[1]) if _ > 0]
        ds_gen.task_engineer.eng_golds(ex)

    metrics_interp = ds_gen.task_engineer.eval(ds_gen)

    if dump_path:
        try:
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            ds_gen.snap(out_path=dump_path)
        except Exception as e:
            print(f"[dump] post-edit prediction snapshot failed for {dump_path}: {e}", flush=True)

    return metrics_interp, metrics_edit