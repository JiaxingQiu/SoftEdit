from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from .utils import *
from .utils.data import RANKED_DATASETS, load_ranked_parquet
import pandas as pd

class VQADataset(Dataset):
    _logged_soft_gen_filter = set()  # dataset_names already logged (avoid repeat spam)

    def __init__(self, config):
        self.config = config
        self.data = []
        df = self.load_df()
        self.data = self.df2data(df)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def load_df(self):
        dataset_name = self.config.experiment.dataset_name
        if dataset_name not in RANKED_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {sorted(RANKED_DATASETS)}")
        df = load_ranked_parquet(dataset_name)
        df["uid"] = df["uid"].astype(str)
        # Default: keep only samples with soft-generality images available
        if "soft_gen_available" in df.columns:
            n_before = len(df)
            df = df[df["soft_gen_available"].astype(bool)].reset_index(drop=True)
            if dataset_name not in VQADataset._logged_soft_gen_filter:
                print(
                    f"[VQADataset] soft_gen_available filter: {len(df)}/{n_before} "
                    f"({dataset_name})",
                    flush=True,
                )
                VQADataset._logged_soft_gen_filter.add(dataset_name)
        return df
    
    def df2data(self, df: pd.DataFrame) -> List[Dict]:
        if df.empty:
            return []
        cols = ["uid", "image_path", "question", "answer_counts"]
        records = df[cols].to_dict(orient="records")
        examples: List[Dict] = []
        for r in records:
            counts = json.loads(r["answer_counts"]) if isinstance(r["answer_counts"], str) else r["answer_counts"]
            ranked = [label for label, _ in sorted(counts.items(), key=lambda x: -x[1]) if _ > 0]
            ex = {
                "uid": str(r["uid"]),
                "image": r["image_path"],
                "question": r["question"],
                "answer_counts": counts,
                "labels_ranked": ranked,
            }
            examples.append(ex)
        return examples

    def set_dataloader(self, batch_size=None, **kwargs):
        task = self.config.experiment.task
        # Allow callers to override the batch size (e.g. sequential editing edits
        # one sample at a time); default to the configured batch size.
        batch_size = batch_size if batch_size is not None else self.config.batch_size
        seed = self.config.seed
        model_name = getattr(self.config.model, "name", "")
        kwargs.setdefault("label_strategy", getattr(self.config.experiment, "label_strategy", "text"))

        self.task_engineer = get_taskengineer(task, seed=seed, model_name=model_name, **kwargs)
        for i, ex in enumerate(self.data):
            ex['idx'] = i
            self.task_engineer.eng_golds(ex)
            self.task_engineer.eng_prompt(ex)
        self.loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=self.image_collate)
        
    def _resize_image(self, img, max_side=800):
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            s = max_side / m
            img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
        return img
    
    def image_collate(self, batch):
        """Collate function that loads images and returns a batch dict.
        Expects items with keys: 'image' (path), 'prompt' (string), 'gold' (dict), 'idx' (int).
        """
        images = []
        prompts = []
        golds = []
        idxs = []
        for ex in batch:
            img_path = ex["image"]
            try:
                img = self._resize_image(Image.open(img_path).convert("RGB"))
            except Exception as exc: 
                # Log the error type and offending image path, then skip this example.
                print(f"[image_collate] Skipping bad image ({exc.__class__.__name__}): {img_path}. Error: {exc}", flush=True)
                continue
            images.append(img)
            prompts.append(ex["prompt"])
            golds.append(ex["gold"])
            idxs.append(ex["idx"])
        return {
            "images": images,
            "prompts": prompts,
            "golds": golds,
            "idxs": idxs,
        }

    # def task_generate_batch(self, batch, model):
    #     """Generate predictions for a single collated batch and write back in place using indices."""
    #     outs = model.generate(batch["images"], batch["prompts"], max_new_tokens=100)
    #     for idx, a in zip(batch["idxs"], outs):
    #         self.task_engineer.eng_preds(self.data[idx], a, model)

    def task_generate(self, model, use_cache=True, max_new_tokens=None, compute_tf=None):
        if hasattr(self.task_engineer, 'run_generate'):
            kwargs = dict(use_cache=use_cache, max_new_tokens=max_new_tokens)
            if compute_tf is not None:
                kwargs["compute_tf"] = compute_tf
            self.task_engineer.run_generate(self, model, **kwargs)
            return
        if max_new_tokens is None:
            max_new_tokens = getattr(self.config, "max_new_tokens", 10)
        for batch in self.loader:
            # Skip empty batches (all images failed to load, e.g., corrupted/truncated images)
            if not batch["images"] or not batch["prompts"]:
                continue
            try:
                core = getattr(getattr(model, "model", model), "model", getattr(model, "model", model))
                cfg = getattr(core, "config", None)
                if cfg is not None:
                    cfg.use_cache = use_cache
                # Qwen3-VL: reset rope deltas Disable gradient checkpointing if available, (owner can be core or core.model)
                if hasattr(core, "gradient_checkpointing_disable"): 
                    core.gradient_checkpointing_disable()
                rope_owner = core if hasattr(core, "rope_deltas") else getattr(core, "model", None) 
                if rope_owner is not None and hasattr(rope_owner, "rope_deltas"):
                    rope_owner.rope_deltas = None
            except Exception:
                pass
            outs = model.generate(batch["images"], batch["prompts"], max_new_tokens=max_new_tokens, use_cache=use_cache)
            for idx, a in zip(batch["idxs"], outs):
                self.task_engineer.eng_preds(self.data[idx], a, model)
    
    def reparse_snapshot(self, data=None):
        """Re-derive parsed pred labels for a loaded prediction snapshot using this
        dataset's task engineer, honoring the current parsing / label_strategy.

        The engineer is created in ``set_dataloader``; on the snapshot-load path
        that hasn't run yet, so build it first. Operates on ``self.data`` by
        default, or an explicit ``data`` list (e.g. baseline's resume buffer).
        """
        if getattr(self, "task_engineer", None) is None:
            self.set_dataloader()
        self.task_engineer.reparse(self.data if data is None else data)

    def get_edits(self):
        """Edit set = samples with rank_acc == 0 (pred not among the acceptable labels).

        Uses rank_acc (tier-position) rather than soft_acc: on count-based datasets
        rank_acc == 0 <=> soft_acc == 0 (both mean the prediction is absent from the
        human answers), so this is behavior-identical here. But rank_acc needs only the
        label *ranking* (no annotator counts), so the same criterion also works on
        ranking-only datasets where soft_acc is undefined. Reads the already-parsed
        `pred["label"]`; the label strategy is applied upstream at generation / load.
        """
        from revlm.metrics.ranked_metrics import build_labels_true, rank_acc

        print("getting edits (rank_acc == 0) from parsed pred label")
        for ex in self.data:
            pred = ex.get("pred") or ex.get("pred_qa") or {}
            pred_label = str(pred.get("label", "")).strip().lower()
            gold = ex.get("gold", {})
            answer_counts = gold.get("answer_counts") or ex.get("answer_counts", {})
            labels_true = build_labels_true(answer_counts)
            ex["edit"] = rank_acc(pred_label, labels_true) == 0.0
        edit_ds = self
        edit_ds.data = [ex for ex in edit_ds.data if ex["edit"]]
        edit_ds.set_dataloader()
        return edit_ds

    # Runtime-only keys that should not be persisted to disk.
    _RUNTIME_KEYS = {"idx", "scores"}

    def snap(self, out_path=None) -> None:
        if out_path is None:
            out_path = os.path.join(self.config.pred_dir, self.config.fname)
        clean = [{k: v for k, v in ex.items() if k not in self._RUNTIME_KEYS} for ex in self.data]
        with open(out_path, "w") as f:
            json.dump(clean, f, indent=2)

    def task_eval(self) -> None:
        task_metrics = self.task_engineer.eval(self)
        out_path = os.path.join(self.config.task_dir, self.config.fname)
        with open(out_path, "w") as f:
            json.dump(task_metrics, f, indent=2)
        print(f"Saved task evaluation metrics to {out_path}")
        print(f"Task evaluation metrics: {task_metrics}", flush=True)