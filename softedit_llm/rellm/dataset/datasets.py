from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import json
import os
from .utils import *
from .utils.data import RANKED_DATASETS, load_ranked_parquet
import pandas as pd


class TQADataset(Dataset):
    """Text QA dataset with ranked human-uncertainty labels (LLM analog of VQADataset).

    Each example carries a question and an ``answer_counts`` distribution over the
    top-k human answer clusters; no images. Prompts / golds / preds are engineered
    by the task engineer (see ``utils/taskengineer.py``).
    """

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
        return df

    def df2data(self, df: pd.DataFrame) -> List[Dict]:
        if df.empty:
            return []
        cols = ["uid", "question", "answer_counts"]
        records = df[cols].to_dict(orient="records")
        examples: List[Dict] = []
        for r in records:
            counts = json.loads(r["answer_counts"]) if isinstance(r["answer_counts"], str) else r["answer_counts"]
            ranked = [label for label, c in sorted(counts.items(), key=lambda x: -x[1]) if c > 0]
            ex = {
                "uid": str(r["uid"]),
                "question": r["question"],
                "answer_counts": counts,
                "labels_ranked": ranked,
            }
            examples.append(ex)
        return examples

    def set_dataloader(self, batch_size=None, **kwargs):
        task = self.config.experiment.task
        batch_size = batch_size if batch_size is not None else self.config.batch_size
        seed = self.config.seed
        model_name = getattr(self.config.model, "name", "")
        kwargs.setdefault("label_strategy", getattr(self.config.experiment, "label_strategy", "text"))

        self.task_engineer = get_taskengineer(task, seed=seed, model_name=model_name, **kwargs)
        for i, ex in enumerate(self.data):
            ex['idx'] = i
            self.task_engineer.eng_golds(ex)
            self.task_engineer.eng_prompt(ex)
        self.loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=0,
                                 pin_memory=True, collate_fn=self.text_collate)

    def text_collate(self, batch):
        """Collate function returning a batch dict of text prompts.

        Expects items with keys: 'prompt' (string), 'gold' (dict), 'idx' (int).
        """
        prompts = []
        golds = []
        idxs = []
        questions = []
        for ex in batch:
            prompts.append(ex["prompt"])
            golds.append(ex["gold"])
            idxs.append(ex["idx"])
            # Raw question (pre-instruction/template) — used by AlphaEdit to key
            # on the last question-content token instead of the fixed suffix.
            questions.append(ex.get("question", ""))
        return {
            "prompts": prompts,
            "golds": golds,
            "idxs": idxs,
            "questions": questions,
        }

    def task_generate(self, model, use_cache=True, max_new_tokens=None, compute_tf=None):
        if hasattr(self.task_engineer, 'run_generate'):
            kwargs = dict(use_cache=use_cache, max_new_tokens=max_new_tokens)
            if compute_tf is not None:
                kwargs["compute_tf"] = compute_tf
            self.task_engineer.run_generate(self, model, **kwargs)
            return
        if max_new_tokens is None:
            max_new_tokens = getattr(self.config, "max_new_tokens", 32)
        for batch in self.loader:
            if not batch["prompts"]:
                continue
            try:
                core = getattr(model, "model", model)
                cfg = getattr(core, "config", None)
                if cfg is not None:
                    cfg.use_cache = use_cache
                if hasattr(core, "gradient_checkpointing_disable"):
                    core.gradient_checkpointing_disable()
            except Exception:
                pass
            outs = model.generate(batch["prompts"], max_new_tokens=max_new_tokens, use_cache=use_cache)
            for idx, a in zip(batch["idxs"], outs):
                self.task_engineer.eng_preds(self.data[idx], a, model)

    def reparse_snapshot(self, data=None):
        """Re-derive parsed pred labels for a loaded prediction snapshot using this
        dataset's task engineer, honoring the current parsing / label_strategy.
        """
        if getattr(self, "task_engineer", None) is None:
            self.set_dataloader()
        self.task_engineer.reparse(self.data if data is None else data)

    def get_edits(self):
        """Edit set = samples with rank_acc == 0 (pred not among the acceptable labels).

        Reads the already-parsed ``pred["label"]``; the label strategy is applied
        upstream at generation / load.
        """
        from rellm.metrics.ranked_metrics import build_labels_true, rank_acc

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
